"""

Link failure handling (Additional DS etc)

This component performs the relabeling  (Instead of our custom discovery)
1. Access to packetIn (switch) and flows (MAIN REASON)
2. Network graph updates rare
"""

# These next two imports are common POX convention
from collections import defaultdict
from copy import deepcopy
import time
from numpy import block
from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.recoco import Timer, Sleep
from pox.lib.revent import EventHalt
from pox.lib.addresses import EthAddr
from pox.lib.packet.icmp import TYPE_ECHO_REPLY
from pox.lib.packet.ipv4 import ipv4 
from pox.lib.packet.ethernet import ethernet, ethtype_to_str
from pox.lib.util import dpid_to_str, str_to_dpid

from os import link
from controller_constants import WARMUP_PERIOD
from librelabeling import find_feasible_path, run_relabel_heuristic
from controller_constants import  SWITCH_MAC_TO_DPID, LINKS, SWITCHES, RELABELING_PERIOD, REBOOT_PERIOD, HOSTS
from controller_constants import flows_level_method, min_sec_lvl, max_sec_lvl, Flow, flow_weight, MLSLink
from random import randint


# Even a simple usage of the logger is much nicer than print!
log = core.getLogger()

# To send out all ports, we can use either of the special ports
# OFPP_FLOOD or OFPP_ALL.  We'd like to just use OFPP_FLOOD,
# but it's not clear if all switches support this, so we make
# it selectable.
all_ports = of.OFPP_FLOOD


class l2_pairs_mls(object):
    def __init__(self):
        core.openflow.addListeners(self)
        core.listen_to_dependencies(self)
        # This table maps (switch,MAC-addr) pairs to the port on 'switch' at
        # which we last saw a packet *from* 'MAC-addr'.
        # (In this case, we use a Connection object for the switch.)
        self.table = {}

        # Map mac address to (switch,port) - Need this for host
        self.mac_table = {}

        self.network_graph = None

        # To generate flow ids
        self.flow_id = 0
        # To uniquely identify a flow  - Source mac, dest mac for now (Key for these dicts below) to flow object
        # Seems like without level info in packet header (flow rules would allow all level flows to go through from source-> dst because rule does not use level info while matching)
        self.flows={}
        self.active_flows = {}
        self.blocked_flows = {}
        self.waiting_flows={}

        # DS for link failure
        self.failed_links=set()
        # To track flows over failed links - Link key to flows through it (if the flow was active and used this path)
        self.link_keys_to_flows=defaultdict(set)
    
        # Rebooting switches (DPID)
        self.rebooting_switches = set()

        self.is_warmup = True
        Timer(WARMUP_PERIOD,
              self._warmup_complete_handler, recurring=False)
        
    def clear_tables_on_all_switches(self):
        msg = of.ofp_flow_mod(command=of.OFPFC_DELETE)
        # iterate over all connected switches and delete all their flows
        for connection in core.openflow.connections:  # _connections.values() before betta
            connection.send(msg)
        log.debug("Clearing all flows from %s swicthes.",
                  len(core.openflow.connections))

    def _warmup_complete_handler(self):
        log.debug("Warm up complete. Host MAC addresses learned %s:",
                  len(self.mac_table.keys()))
        # print(self.mac_table.keys())
        core.openflow_discovery.set_host_info(self.mac_table)
        self.network_graph = core.openflow_discovery.get_network_graph

        # Clear all flows
        self.clear_tables_on_all_switches()
        self.is_warmup = False

        # Coverage stats timer 
        Timer(1,self._record_coverage_stats,recurring=True)   

        # After warmup is done we start the periodic relabeling
        Timer(RELABELING_PERIOD, self._perform_relabel, recurring=True)

    def _handle_openflow_discovery_LinkEvent(self, event):
        # Updates view of network when links change (CHeck if this is the correct handler for link failure events)
        if not self.is_warmup:
            if event.removed:
                # Update our network graph ds (Link failed so only link ds)
                source_dpid=dpid_to_str(event.link.dpid1)
                dest_dpid=dpid_to_str(event.link.dpid2)  
                # If it's already been failed - assume that it also dealt with relevant flows
                if event.link in self.failed_links:
                    return EventHalt
                else:     
                    log.debug("Link might have failed")
                    if (source_dpid,dest_dpid) in self.network_graph[LINKS]:          
                        link_obj=self.network_graph[LINKS].pop((source_dpid,dest_dpid))
                    if (dest_dpid,source_dpid) in self.network_graph[LINKS]:          
                        link_obj=self.network_graph[LINKS].pop((dest_dpid,source_dpid))  
                    self.failed_links.add(event.link) 
             
                    # What about existing active flows which were using this link? Should we update the DS here?
                    for flow_obj in self.link_keys_to_flows[(source_dpid,dest_dpid)]:
                        if (flow_obj.source,flow_obj.dest) in self.active_flows:
                            self.active_flows.pop((flow_obj.source,flow_obj.dest))
                        self.blocked_flows[(flow_obj.source,flow_obj.dest)]=flow_obj

                    for flow_obj in self.link_keys_to_flows[(dest_dpid,source_dpid)]:
                        if (flow_obj.source,flow_obj.dest) in self.active_flows:
                            self.active_flows.pop((flow_obj.source,flow_obj.dest))
                        self.blocked_flows[(flow_obj.source,flow_obj.dest)]=flow_obj  
            elif event.added:
                if event.link in self.failed_links:
                    log.debug("Link might have been added")
                    source_dpid = dpid_to_str(event.link.dpid1)
                    source_obj=self.network_graph[SWITCHES][source_dpid]
                    source_port=event.link.port1
                    dest_dpid = dpid_to_str(event.link.dpid2)
                    dest_obj=self.network_graph[SWITCHES][dest_dpid]
                    dest_port=event.link.port2
                    # Update our internal ds
                    if (source_dpid,dest_dpid) not in self.network_graph[LINKS].keys():
                        self.network_graph[LINKS][source_dpid,dest_dpid] = MLSLink(source_obj, dest_obj,source_port=source_port,dest_port=dest_port)
                        self.network_graph[LINKS][(dest_dpid,source_dpid)] = MLSLink(dest_obj, source_obj,source_port=dest_port,dest_port=source_port)
                    self.failed_links.remove(event.link)  
            return EventHalt      
               
    def install_rules_for_flow(self, ofp_packet_in, path,flow_obj):
        """
            Iterates through path backwards (can also do forward**) and on each switch in path installs necessary rule
            flow: Flow object as defined in controller common
            packet_in: To forward the packet
            path: list of nodes
            flow_obj - Needed for link failure
            ** - Would have to install rules on all hops first and then forward actual packet
        """
        path_len=len(path)
        links_dict=self.network_graph[LINKS]
        switches_dict=self.network_graph[SWITCHES]

        # We want to install rules & forward this actual packet as well (only to first hop because all the rules are installed before forwarding)
        # To do this we reverse path and install rules backwards
        path=path[::-1]
        
        # Traversing the path backwards and installing rules
        for pos in range(path_len):
            curr_node=path[pos]
            if curr_node in switches_dict:
                # Because we're traversing the path backwards
                prev_node=path[pos+1]
                next_node=path[pos-1]

                if (prev_node,curr_node) not in links_dict:
                    self.active_flows.pop((flow_obj.source,flow_obj.dest))
                    self.blocked_flows[(flow_obj.source,flow_obj.dest)]=flow_obj
                    return
                
                if (curr_node,next_node) not in links_dict:
                    self.active_flows.pop((flow_obj.source,flow_obj.dest))
                    self.blocked_flows[(flow_obj.source,flow_obj.dest)]=flow_obj
                    return    

                # Links-flows mapping (link failure)    
                self.link_keys_to_flows[(prev_node,curr_node)].add(flow_obj)

                in_link_to_curr_node=links_dict[(prev_node,curr_node)]
                out_link_from_curr_node=links_dict[(curr_node,next_node)]
                # We want the port on the curr node (dst port of the in link)
                in_port=in_link_to_curr_node.dest_port
                # We want the port on the curr node (src port of the out link)
                dst_port=out_link_from_curr_node.source_port

                # Install rule
                msg = of.ofp_flow_mod()
                # Forward the incoming packet (first hop)
                if pos==path_len-2:
                    msg.data = ofp_packet_in
                # Should we add in port? They don't (L2 PAIRS)
                # Convert MAC string back to EthAddr
                # Source 
                msg.match.dl_src = EthAddr(path[-1])
                # Dest
                msg.match.dl_dst = EthAddr(path[0])
                msg.actions.append(of.ofp_action_output(port=dst_port))
                core.openflow.sendToDPID(dpid=str_to_dpid(curr_node),data=msg)  

        # log.info("Flow rules installed %s->%s",path[-1],path[0])            

    def reboot_switches(self,switches_need_to_be_rebooted):
        """
            Mimic rebooting 
            Can try sleep if there are separate threads or processes (Need to change design similar to l2_learning)
            Instead we just effectively get the rebooting switches to do nothing for the REBOOT_PERIOD through a timer
        """
        self.rebooting_switches=switches_need_to_be_rebooted
        Timer(REBOOT_PERIOD,self._reboot_finish, recurring=False,args=[switches_need_to_be_rebooted])
        
    def _reboot_finish(self,switches_finished_rebooting):
        self.rebooting_switches.clear()
        # Clear the waiting flows. If they departed this is correct, if not we map it back to the flow object and install rules (if not installed already)
        self.waiting_flows.clear()    


    def _use_normal_forwaring(self,event):
        """
            Use l2 pairs default
        """
        packet = event.parsed

        # Learn the source
        self.table[(event.connection, packet.src)] = event.port

        dst_port = self.table.get((event.connection, packet.dst))

        source_addr = packet.src.toStr()
        dest_addr = packet.dst.toStr()

        if dst_port is None:
            # We don't know where the destination is yet.  So, we'll just
            # send the packet out all ports (except the one it came in on!)
            # and hope the destination is out there somewhere. :)
            msg = of.ofp_packet_out(data=event.ofp)
            msg.actions.append(of.ofp_action_output(port=all_ports))
            event.connection.send(msg)


        else:
            # Since we know the switch ports for both the source and dest
            # MACs, we can install rules for both directions.
            if not self.is_warmup:
                msg = of.ofp_flow_mod()
                msg.match.dl_dst = packet.src
                msg.match.dl_src = packet.dst
                msg.actions.append(of.ofp_action_output(port=event.port))
                event.connection.send(msg)

            # This is the packet that just came in -- we want to
            # install the rule and also resend the packet.
            msg = of.ofp_flow_mod()
            msg.data = event.ofp  # Forward the incoming packet
            msg.match.dl_src = packet.src
            msg.match.dl_dst = packet.dst
            msg.actions.append(of.ofp_action_output(port=dst_port))
            event.connection.send(msg)

            # We record hosts and host links here under the assumption that discovery is over with pingall (flooding)
            if self.is_warmup and source_addr not in self.mac_table:
                # log.debug("Edge port(CONFIRM): %s %s, src: %s", dpid_to_str(
                #     event.connection.dpid), str(event.port), source_addr)

                self.mac_table[source_addr] = (
                    dpid_to_str(event.connection.dpid), event.port)
            

        
    # Handle messages the switch has sent us because it has no
    # matching rule.
    def _handle_PacketIn(self, event):
        packet = event.parsed

        # Learn the source
        self.table[(event.connection, packet.src)] = event.port

        dst_port = self.table.get((event.connection, packet.dst))

        source_addr = packet.src.toStr()
        dest_addr = packet.dst.toStr()

        # If warmup (Learning topology) then normal routing or ARP (Can deal with later) or IPV6 (Unsure about this)
        if self.is_warmup or packet.type==ethernet.IPV6_TYPE or packet.type==ethernet.ARP_TYPE:
            self._use_normal_forwaring(event)
            
        # New flows or blocked/waiting flows would come here. Active flows should come here only once
        else:

            # Blocked
            if (source_addr, dest_addr) in self.blocked_flows:
                # TODO - Add drop rules for blocked flows??
                # flow_level=self.blocked_flows[(source_addr, dest_addr)].level
                # log.debug("Flow %s->%s type: %s, level: %s is already blocked",source_addr,dest_addr,ethtype_to_str(packet.type),str(flow_level))
                return


            elif (source_addr,dest_addr) in self.flows:    
                flow_obj=self.flows[(source_addr, dest_addr)]
      
            else:

                # We don't care about ICMP response also we shouldn't mistakenly create a new flow for these and try routing that
                if (dest_addr,source_addr) in self.flows:
                      # Check it if is an ICMP pkt (echo reply)
                      if packet.next.protocol==ipv4.ICMP_PROTOCOL and packet.next.next.type==TYPE_ECHO_REPLY:
                          # Alternatively we could just forward these as well or even add drop rules on all switches (drop echo replies after warmup)
                          # Now allowing ICMP responses to go through expliclty
                          self._use_normal_forwaring(event)

                if flows_level_method==5:
                    flow_source_level=self.network_graph[HOSTS][source_addr].level
                    flow_dest_level=self.network_graph[HOSTS][dest_addr].level
                    # assert flow_source_level<=flow_dest_level and "Invalid flow generated"
                    flow_level=min(flow_source_level,flow_dest_level)

                flow_obj = Flow(self.flow_id, source_addr, dest_addr, flow_level)
                self.flows[(source_addr,dest_addr)]=flow_obj
                self.flow_id = self.flow_id+1
                log.debug("New flow id: %s | %s->%s type: %s, level: %s",flow_obj.key,source_addr,dest_addr,ethtype_to_str(packet.type),str(flow_level))


            # Invokes heuristic through library API (Dijik helper)
            path = find_feasible_path(self.network_graph, flow_obj)

            if not path:
                log.debug("Flow is blocked  %s->%s",source_addr,dest_addr)
                self.blocked_flows[(source_addr, dest_addr)] = flow_obj
            else:
                # Check if it is waiting for a switch to reboot
                for node_key in path:
                    if node_key in self.rebooting_switches:
                        # if (source_addr,dest_addr) not in self.waiting_flows:
                        #     # log.debug("Flow waiting %s->%s for %s" ,source_addr,dest_addr,node_key)
                        #     self.waiting_flows[(source_addr,dest_addr)]=flow_obj
                        # Call coop Sleep
                        Sleep(timeToWake=REBOOT_PERIOD)
                        break
                        # Let's try the good ol return again with such a large duration we expect that the flow will persist
                        # return
                     

                # Update waiting flows DS if needed
                if (source_addr,dest_addr) in self.waiting_flows:
                    self.waiting_flows.pop((source_addr,dest_addr))

                # Install rule/rules using path found
                # log.debug("Path found %s",str(path))
                self.active_flows[(source_addr, dest_addr)] = flow_obj
                # If over failed link this will update active flows
                self.install_rules_for_flow(event.ofp,path,flow_obj)

                # log.debug("Packet In - Active flows: %s, blocked %s and waiting %s",len(self.active_flows),len(self.blocked_flows),len(self.waiting_flows))    
                # self._record_coverage_stats()

    def _perform_relabel(self):
        # self._record_coverage_stats()
        log.debug("Pre relabel -Active flows: %s, blocked %s and waiting %s",len(self.active_flows),len(self.blocked_flows),len(self.waiting_flows))    
        log.debug("Relabeling now")
        # Step 1 - Clear all rules (This can be optimized later)
        self.clear_tables_on_all_switches()
        # Clear link keys to flows (because new routes)
        self.link_keys_to_flows.clear()

        # Step 2 - Invoke heuristic (LIB API). Should update network graph (switches) as needed in place. No need to add waiting flows but to be generic we've added it
        flows=set(self.active_flows.values())|set(self.blocked_flows.values())|set(self.waiting_flows.values())
        self.active_flows,self.waiting_flows,self.blocked_flows,switches_need_to_be_rebooted=run_relabel_heuristic(self.network_graph,flows)
    
        # Step 3 - Reboot necessary switches 
        # self.rebooting_switches=self.rebooting_switches+switches_need_to_be_rebooted
        self.reboot_switches(switches_need_to_be_rebooted)        
        log.debug("Relabeling finished: %s",len(switches_need_to_be_rebooted))

        # for sw_key in switches_need_to_be_rebooted:
        #     log.debug("Switch (level changed) %s level %s",sw_key,self.network_graph[SWITCHES][sw_key].level)

        log.debug("Post relabel -Active flows: %s, blocked %s and waiting %s",len(self.active_flows),len(self.blocked_flows),len(self.waiting_flows))    
        # self._record_coverage_stats()

    def _record_coverage_stats(self):
        total_flows=len(self.active_flows)+len(self.waiting_flows)+len(self.blocked_flows)
        
        if total_flows==0:
            return 
        coverage=len(self.active_flows)/total_flows

        # log.debug("Coverage: %s",coverage)
        with open("coverage.txt", "a") as coverage_results:
            # Writing data to a file
            coverage_results.write(str(coverage)+"\n")

       # Record objective value
        max_objective_value = 0
        current_objective_value=0
        for flow_obj in self.active_flows.values():
            max_objective_value += flow_weight(flow_obj.level)
            current_objective_value+=flow_weight(flow_obj.level)
        for flow_obj in self.blocked_flows.values():
            max_objective_value += flow_weight(flow_obj.level)
        for flow_obj in self.waiting_flows.values():
            max_objective_value += flow_weight(flow_obj.level)

        with open("objective_value.txt", "a") as objective_value:
            # Writing data to a file
            objective_value.write(str(current_objective_value * 100.0 / max_objective_value)+"\n")    

    

      


def launch():

    # core.openflow.addListenerByName("PacketIn", _handle_PacketIn)

    core.registerNew(l2_pairs_mls)

    log.info("Pair-Learning-MLS switch running.")
