from common import DEBUG, switch_cap, agg_to_core_link_cap, edge_to_agg_link_cap, host_to_edge_link_cap
from common import const_switch_lvl, max_sec_lvl, min_sec_lvl, const_host_lvl, delta_j
from common import hosts_level_method, switches_level_method
from common import hosts_per_wan_switch, hosts_per_mesh_switch, mesh_switch_link_cap

from mininet.topo import Topo
from mininet.link import TCLink
#import networkx as nx
from random import randint, shuffle
from collections import namedtuple


"""
This function creates objects (using named tuples) with  whatever desired attributes were added to switches,hosts and links 
It then updates the topo's interal dictionaries (_switches,_hosts,_links)
"""


def refine_topo_internal_data_structures(topo):
    # Can add sanity checks later

    switch_label=list(topo._switches.keys())[0]
    topo._switches[switch_label]["label"]=switch_label
    host_label=list(topo._hosts.keys())[0]
    topo._hosts[host_label]["label"]=host_label
    link_label=list(topo._links.keys())[0]
    topo._links[link_label]["label"]=link_label

    Switch = namedtuple("Switch", list(
        list(topo._switches.values())[0].keys()))
    Host = namedtuple("Host", list(list(topo._hosts.values())[0].keys()))
    Link = namedtuple("Link", list(list(topo._links.values())[0].keys()))

    for switch_label in topo._switches:
        # Add the label
        topo._switches[switch_label]["label"]=switch_label
        topo._switches[switch_label] = Switch(**topo._switches[switch_label])

    for host_label in topo._hosts:
        # Add the label
        topo._hosts[host_label]["label"]=host_label
        topo._hosts[host_label] = Host(**topo._hosts[host_label])

    for link_label in topo._links:
        # Add the label
        topo._links[link_label]["label"]=link_label        
        topo._links[link_label] = Link(**topo._links[link_label])

    if DEBUG:
        print(topo._switches, topo._links, topo._hosts)


"""
We construct topos with three internal DS (names to dict of desired properties created automatically)
_switches
_hosts
_links
"""


class SimpleTopo(Topo):
    "Simple 2 switch topo connected to n hosts."

    def build(self, n=2):
        # Dict of names to custom objects (which contain properties we passed and access methods)
        self._switches = dict()
        self._hosts = dict()
        self._links = dict()

        self.addSwitch(
            "s1", level=1, capacity=100)
        self._switches["s1"] = self.nodeInfo("s1")
        # Python's range(N) generates 0..N-1
        for h in range(n):
            host_id = 's1_h%s' % (h + 1)
            self.addHost(host_id, level=2)
            self._hosts[host_id] = self.nodeInfo(host_id)
            link_key = (host_id, "s1")

            self.addLink(
                node1=link_key[0], node2=link_key[1], cls=TCLink, bw=host_to_edge_link_cap)
            self._links[link_key] = self.linkInfo(link_key[0], link_key[1])

        self.addSwitch("s2", level=1, capacity=100)
        self._switches["s2"] = self.nodeInfo("s2")

        # Python's range(N) generates 0..N-1
        for h in range(n):
            host_id = 's2_h%s' % (h + 1)
            self.addHost(host_id, level=3)
            self._hosts[host_id] = self.nodeInfo(host_id)
            link_key = (host_id, "s2")
            self.addLink(
                node1=link_key[0], node2=link_key[1], cls=TCLink, bw=host_to_edge_link_cap)
            self._links[link_key] = self.linkInfo(link_key[0], link_key[1])

        link_key = ("s1", "s2")
        self.addLink(
            node1=link_key[0], node2=link_key[1], cls=TCLink, bw=host_to_edge_link_cap)
        self._links[link_key] = self.linkInfo(link_key[0], link_key[1])
        refine_topo_internal_data_structures(self)


class FatTreeTopo(Topo):
    """Generates a fat-tree topology with specified port density (k) per switch.
    Citation: Mohammad Al-Fares, Alexander Loukissas, and Amin Vahdat. 2008. A scalable, commodity data center network architecture. SIGCOMM Comput. Commun. Rev. 38, 4 (October 2008), 63–74. DOI:https://doi.org/10.1145/1402946.1402967.
    Link: http://ccr.sigcomm.org/online/files/p63-alfares.pdf.
    """

    def build(self, n=4):

        # Dict from name to object (properties)
        self._switches = dict()
        self._hosts = dict()
        self._links = dict()

        num_core_sw = (n / 2) ** 2
        num_edge_sw = 2 * ((n / 2) ** 2)
        num_total_sw = 5 * ((n / 2) ** 2)
        num_agg_sw = num_edge_sw
        num_hosts = 2 * ((n / 2) ** 3)
        pod_width = n / 2
        stride_length = pod_width
        num_pods = n

        max_core_lvl_for_this_pod_switch = {core_start_idx: 0 for core_start_idx in range(
            0, int(pod_width) * int(num_pods), int(stride_length))}
        max_agg_lvl_for_pod = {pod_idx: 0 for pod_idx in range(int(num_pods))}

        for pod_idx in range(int(num_pods)):
            sw_start_id_in_pod = pod_idx * pod_width
            for pod_pos in range(int(pod_width)):
                sw_idx = sw_start_id_in_pod + pod_pos
                core_start_idx = pod_pos * stride_length
                # for each of k/2 ports on the switch; only depends on pos in pod

                # add uplink core switches
                for port in range(int(stride_length)):
                    core_switch_label = 's_cor0' + str(int(core_start_idx + port))
                    if not (core_switch_label in self._switches):
                        if switches_level_method == 1:  # fixed
                            switch_lvl = const_switch_lvl
                        else:  # cover method 2 and 3 here
                            switch_lvl = randint(min_sec_lvl, max_sec_lvl)

                        self.addSwitch(name=core_switch_label,
                                       level=switch_lvl, capacity=switch_cap)

                        self._switches[core_switch_label] = self.nodeInfo(
                            core_switch_label)
                        if DEBUG:
                            print('core_switch_lvl: ', switch_lvl)
                        if switch_lvl > max_core_lvl_for_this_pod_switch[core_start_idx]:
                            max_core_lvl_for_this_pod_switch[core_start_idx] = switch_lvl
                        if DEBUG:
                            print('max_core_lvl_for_this_pod_switch[core_start_idx:%d]: %d' % (
                                core_start_idx, max_core_lvl_for_this_pod_switch[core_start_idx]))
                    else:
                        if DEBUG:
                            print("already added core switch")

                # add the agg switches
                for port in range(int(stride_length)):
                    asw_switch_label = 's_asw1' + str(int(sw_idx))
                    if DEBUG:
                        print("max_core_lvl_for_this_pod_switch[core_start_idx:%d]: %d" % (
                            core_start_idx, max_core_lvl_for_this_pod_switch[core_start_idx]))
                    if not (asw_switch_label in self._switches):
                        if switches_level_method == 1:  # fixed
                            switch_lvl = const_switch_lvl
                        elif switches_level_method == 2:  # random
                            switch_lvl = randint(min_sec_lvl, max_sec_lvl)
                        elif switches_level_method == 3:  # other
                            switch_lvl = min(1 + (pod_idx + pod_pos) %
                                             max_sec_lvl, max_core_lvl_for_this_pod_switch[core_start_idx])
                            if switch_lvl == 0:
                                if DEBUG:
                                    print("stopped at 2")
                        self.addSwitch(
                            name=asw_switch_label, level=switch_lvl, capacity=switch_cap)

                        self._switches[asw_switch_label] = self.nodeInfo(
                            asw_switch_label)
                        if switch_lvl > max_agg_lvl_for_pod[pod_idx]:
                            max_agg_lvl_for_pod[pod_idx] = switch_lvl
                    else:
                        if DEBUG:
                            print("already added agg switch")

                # add links from core to agg
                for port in range(int(stride_length)):
                    core_switch_label = 's_cor0' + str(int(core_start_idx + port))
                    asw_switch_label = 's_asw1' + str(int(sw_idx))
                    link_key = (core_switch_label, asw_switch_label)
                    if link_key not in self._links.keys():
                        self.addLink(node1=link_key[0], node2=link_key[1], cls=TCLink,
                                     bw=agg_to_core_link_cap)
                        self._links[link_key] = self.linkInfo(
                            link_key[0], link_key[1])

                # add the agg switches
                for port in range(int(stride_length)):
                    asw_switch_label = 's_asw1' + \
                        str(int(sw_start_id_in_pod + port))
                    if DEBUG:
                        print("max_core_lvl_for_this_pod_switch[core_start_idx:%d]: %d" % (
                            core_start_idx, max_core_lvl_for_this_pod_switch[core_start_idx]))
                    if not (asw_switch_label in self._switches):
                        if switches_level_method == 1:  # fixed
                            switch_lvl = const_switch_lvl
                        elif switches_level_method == 2:  # random
                            switch_lvl = randint(min_sec_lvl, max_sec_lvl)
                        elif switches_level_method == 3:  # other
                            switch_lvl = min(1 + (pod_idx + pod_pos) %
                                             max_sec_lvl, max_core_lvl_for_this_pod_switch[core_start_idx])
                            if switch_lvl == 0:
                                if DEBUG:
                                    print("stopped at 5")
                        self.addSwitch(
                            name=asw_switch_label, level=switch_lvl, capacity=switch_cap)
                        self._switches[asw_switch_label] = self.nodeInfo(
                            asw_switch_label)
                        if switch_lvl > max_agg_lvl_for_pod[pod_idx]:
                            max_agg_lvl_for_pod[pod_idx] = switch_lvl
                    else:
                        if DEBUG:
                            print("already added agg switch5")

                # Add edge switch and link from agg to edge switch
                for port in range(int(stride_length)):
                    edge_switch_label = 's_esw2' + str(int(sw_idx))
                    asw_switch_label = 's_asw1' + \
                        str(int(sw_start_id_in_pod + port))
                    if DEBUG:
                        print("max_agg_lvl_for_pod[pod_idx:%d]: %d" % (
                            pod_idx, max_agg_lvl_for_pod[pod_idx]))
                    if not (edge_switch_label in self._switches):
                        if switches_level_method == 1:  # fixed
                            switch_lvl = const_switch_lvl
                        elif switches_level_method == 2:  # random
                            switch_lvl = randint(min_sec_lvl, max_sec_lvl)
                        elif switches_level_method == 3:  # other
                            switch_lvl = min(1 + (pod_idx + pod_pos) %
                                             max_sec_lvl, max_agg_lvl_for_pod[pod_idx])
                            if switch_lvl == 0:
                                if DEBUG:
                                    print("stopped at 3")
                        self.addSwitch(
                            name=edge_switch_label, level=switch_lvl, capacity=switch_cap)
                        self._switches[edge_switch_label] = self.nodeInfo(
                            edge_switch_label)
                    else:
                        if DEBUG:
                            print("already added edge switch")

                    link_key = (asw_switch_label, edge_switch_label)
                    if link_key not in self._links.keys():
                        self.addLink(
                            link_key[0], link_key[1],
                            cls=TCLink, bw=edge_to_agg_link_cap)
                        self._links[link_key] = self.linkInfo(
                            link_key[0], link_key[1])

                    # if ('asw' + str(int(sw_start_id_in_pod + port)), 'esw' + str(int(sw_idx))) not in self._links.keys():
                    #     self._links[('asw' + str(int(sw_start_id_in_pod + port)), 'esw' + str(int(sw_idx)))] = self.addLink(
                    #         self._switches['asw' +
                    #                     str(int(sw_start_id_in_pod + port))], self._switches['esw' + str(int(sw_idx))],cls=TCLink,
                    #         bw=edge_to_agg_link_cap)

                if DEBUG:
                    print("===")

                subnet_lvl = randint(min_sec_lvl, max_sec_lvl)

                # also add a link to hosts for each port
                for port in range(int(stride_length)):
                    host_label = 'h' + \
                        str(int((sw_idx * stride_length) + port))
                    edge_switch_label = 's_esw2' + str(int(sw_idx))
                    if not (host_label in self._hosts):
                        if hosts_level_method == 1:  # fixed
                            switch_lvl = const_host_lvl  # overload use of switch_lvl for hosts
                        elif hosts_level_method == 2:  # random
                            switch_lvl = randint(min_sec_lvl, max_sec_lvl)
                        elif hosts_level_method == 3:  # other1
                            switch_lvl = randint(max(
                                min_sec_lvl, self._switches[edge_switch_label]['level'] - delta_j),
                                self._switches[edge_switch_label]['level'])
                        elif hosts_level_method == 4:  # other2
                            switch_lvl = min(1 + (pod_idx + pod_pos) % max_sec_lvl,
                                             self._switches[edge_switch_label]['level'])
                        # other3 (same as parent)
                        elif hosts_level_method == 5:
                            switch_lvl = self._switches[edge_switch_label]['level']
                        # other4 (same in subnet)
                        elif hosts_level_method == 6:
                            switch_lvl = subnet_lvl
                        self.addHost(host_label, level=switch_lvl)
                        self._hosts[host_label] = self.nodeInfo(host_label)
                    else:
                        if DEBUG:
                            print("already added host")

                    link_key = (edge_switch_label, host_label)
                    if link_key not in self._links.keys():
                        self.addLink(
                            link_key[0], link_key[1],
                            cls=TCLink, bw=host_to_edge_link_cap)

                        self._links[link_key] = self.linkInfo(
                            link_key[0], link_key[1])

        refine_topo_internal_data_structures(self)

    @property
    def getlinks(self):
        return self._links

class MeshTopo(Topo):
    def build(self, num_switch, fully_connected=False):

        self._switches = dict()
        self._hosts = dict()
        self._links = dict()

        # Generate n number of switches and connect nth switch to hosts
        for s in range(1, num_switch + 1):
            sw_label = "s%s" % s
            if switches_level_method == 1:
                sw_level = const_switch_lvl
            elif switches_level_method == 2:
                sw_level = randint(min_sec_lvl, max_sec_lvl)

            self.addSwitch(sw_label, level=sw_level, capacity=switch_cap)
            self._switches[sw_label] = self.nodeInfo(sw_label)
            host_lvl = 0
            subnet_lvl = randint(min_sec_lvl, max_sec_lvl)

            for h in range(1, hosts_per_mesh_switch + 1):
                host_label = "h%s" % (h + ((s - 1) * hosts_per_mesh_switch))
                if hosts_level_method == 1:  # fixed
                    host_lvl = const_host_lvl
                elif hosts_level_method == 2:  # random
                    host_lvl = randint(min_sec_lvl, max_sec_lvl)
                elif hosts_level_method == 3:  # within delta_j (always valid)
                    host_lvl = randint(
                        max(min_sec_lvl, sw_level - delta_j), sw_level)
                elif hosts_level_method == 4:  # <= sw_lvl
                    # up to sw level (might be too low tho)
                    host_lvl = randint(min_sec_lvl, sw_level)
                elif hosts_level_method == 5:  # == sw_lvl
                    host_lvl = sw_level
                elif hosts_level_method == 6:  # other4 (same in subnet)
                    host_lvl = subnet_lvl

                self.addHost(host_label, level=host_lvl)
                self._hosts[host_label] = self.nodeInfo(host_label)
                # Add switch-host link

                link_label = (sw_label, host_label)
                self.addLink(link_label[0], link_label[1],
                             bw=host_to_edge_link_cap)

                self._links[link_label] = self.linkInfo(
                    link_label[0], link_label[1])

        # Add links between switches
        if fully_connected == True:
            for src_switch in range(1, num_switch + 1):
                for dst_switch in range(src_switch + 1, num_switch + 1):
                    sw_label = "s%s" % src_switch
                    sw_label_1 = "s%s" % dst_switch
                    link_label = (sw_label, sw_label_1)

                    self.addLink(
                        link_label[0], link_label[1], bw=mesh_switch_link_cap)
                    self._links[link_label] = self.linkInfo(
                        link_label[0], link_label[1])

        else:
            for src_switch in range(1, num_switch + 1):
                for dst_switch in range(1, num_switch + 1):
                    # 0.9 probability of link being created
                    if src_switch != dst_switch and randint(1, 100) > 10:
                        sw_label = "s%s" % src_switch
                        sw_label_1 = "s%s" % dst_switch
                        link_label = (sw_label, sw_label_1)

                        self.addLink(
                            link_label[0], link_label[1], bw=mesh_switch_link_cap)
                        self._links[link_label] = self.linkInfo(link_label[0], link_label[1])    
                 

        refine_topo_internal_data_structures(self)

    @property
    def getlinks(self):
        return self._links
class StarWANTopo(Topo):
    def build(self):
        WAN = nx.read_gml("AttNA.gml", label="id")

        self._switches = dict()
        self._hosts = dict()
        self._links = dict()

        for sw_pos in range(WAN.number_of_nodes()):
            sw_label = "sw" + str(sw_pos)
            if switches_level_method == 1:  # fixed
                sw_level = const_switch_lvl
            elif switches_level_method == 2:  # random
                sw_level = randint(min_sec_lvl, max_sec_lvl)
            self.addSwitch(
                sw_label, level=sw_level, capacity=switch_cap, protocols="OpenFlow10")
            self._switches[sw_label] = self.nodeInfo(sw_label)
            host_lvl = 0
            subnet_lvl = randint(min_sec_lvl, max_sec_lvl)
            for host_pos in range(hosts_per_wan_switch):
                host_label = 'h' + \
                    str((sw_pos * hosts_per_wan_switch + host_pos))
                if hosts_level_method == 1:  # fixed
                    host_lvl = const_host_lvl
                elif hosts_level_method == 2:  # random
                    host_lvl = randint(min_sec_lvl, max_sec_lvl)
                elif hosts_level_method == 3:  # within delta_j (always valid)
                    host_lvl = randint(
                        max(min_sec_lvl, sw_level - delta_j), sw_level)
                elif hosts_level_method == 4:  # <= sw_lvl
                    # up to sw level (might be too low tho)
                    host_lvl = randint(min_sec_lvl, sw_level)
                elif hosts_level_method == 5:  # == sw_lvl
                    host_lvl = sw_level
                elif hosts_level_method == 6:  # other4 (same in subnet)
                    host_lvl = subnet_lvl
                self.addHost(
                    host_label, level=host_lvl)
                self._hosts[host_label] = self.nodeInfo(host_label)
                # Add switch-host link
                link_key = (sw_label, host_label)
                self.addLink(node1=link_key[0], node2=link_key[1],
                             cls=TCLink, bw=host_to_edge_link_cap)
                self._links[link_key] = self.linkInfo(link_key[0], link_key[1])

        for edge in WAN.edges():
            sw_from_label = "sw" + str(edge[0])
            sw_dest_label = "sw" + str(edge[1])
            link_key = (sw_from_label, sw_dest_label)
            self.addLink(node1=link_key[0], node2=link_key[1],
                         cls=TCLink, bw=edge_to_agg_link_cap)

            self._links[link_key] = self.linkInfo(link_key[0], link_key[1])

        refine_topo_internal_data_structures(self)
    @property
    def getlinks(self):
        return self._links

# For debugging purposes (Differs in that no extra parameters are passed to switch etc)
# class MeshTopoNoChange(Topo):
#     def build(self, num_switch, fully_connected=False):

#         self._switches = dict()
#         self._hosts = dict()
#         self._links = dict()

#         # Generate n number of switches and connect nth switch to hosts
#         for s in range(1, num_switch + 1):
#             sw_label = "s%s" % s
#             if switches_level_method == 1:
#                 sw_level = const_switch_lvl
#             elif switches_level_method == 2:
#                 sw_level = randint(min_sec_lvl, max_sec_lvl)

#             self.addSwitch(sw_label)
#             self._switches[sw_label] = self.nodeInfo(sw_label)
#             host_lvl = 0
#             subnet_lvl = randint(min_sec_lvl, max_sec_lvl)

#             for h in range(1, hosts_per_mesh_switch + 1):
#                 host_label = "h%s" % (h + ((s - 1) * hosts_per_mesh_switch))
#                 if hosts_level_method == 1:  # fixed
#                     host_lvl = const_host_lvl
#                 elif hosts_level_method == 2:  # random
#                     host_lvl = randint(min_sec_lvl, max_sec_lvl)
#                 elif hosts_level_method == 3:  # within delta_j (always valid)
#                     host_lvl = randint(
#                         max(min_sec_lvl, sw_level - delta_j), sw_level)
#                 elif hosts_level_method == 4:  # <= sw_lvl
#                     # up to sw level (might be too low tho)
#                     host_lvl = randint(min_sec_lvl, sw_level)
#                 elif hosts_level_method == 5:  # == sw_lvl
#                     host_lvl = sw_level
#                 elif hosts_level_method == 6:  # other4 (same in subnet)
#                     host_lvl = subnet_lvl

#                 self.addHost(host_label)
#                 self._hosts[host_label] = self.nodeInfo(host_label)
#                 # Add switch-host link

#                 link_label = (sw_label, host_label)
#                 self.addLink(link_label[0], link_label[1])

#                 self._links[link_label] = self.linkInfo(
#                     link_label[0], link_label[1])

#         # Add links between switches
#         if fully_connected == True:
#             for src_switch in range(1, num_switch + 1):
#                 for dst_switch in range(src_switch + 1, num_switch + 1):
#                     sw_label = "s%s" % src_switch
#                     sw_label_1 = "s%s" % dst_switch
#                     link_label = (sw_label, sw_label_1)

#                     self.addLink(
#                         link_label[0], link_label[1])
#                     self._links[link_label] = self.linkInfo(
#                         link_label[0], link_label[1])

#         else:
#             for src_switch in range(1, num_switch + 1):
#                 for dst_switch in range(1, num_switch + 1):
#                     # 0.9 probability of link being created
#                     if src_switch != dst_switch and randint(1, 100) > 10:
#                         sw_label = "s%s" % src_switch
#                         sw_label_1 = "s%s" % dst_switch
#                         link_label = (sw_label, sw_label_1)

#                         self.addLink(
#                             link_label[0], link_label[1])
#                         self.addLink(
#                             link_label[0], link_label[1])

#         refine_topo_internal_data_structures(self)

# topos= {'MeshTopoNoChange': (lambda : MeshTopoNoChange(5) )}