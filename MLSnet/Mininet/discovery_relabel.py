"""
Custom Discovery-Relabel Component

This module discovers the connectivity between OpenFlow switches by sending
out LLDP packets. To be notified of this information, listen to LinkEvents
on core.openflow_discovery.

This also implements the periodidc relabeling and necessary data structures and methods

This is a modified version of the discovery component so it can be invoked in a similar fashion

We do not expose the graph method (no Discovery graph component)

"""

from copy import deepcopy
from urllib.parse import parse_qs
from pox.lib.revent import *
from pox.lib.recoco import Timer
from pox.lib.util import dpid_to_str, str_to_bool
from pox.core import core
import pox.openflow.libopenflow_01 as of
import pox.lib.packet as pkt
from pox.lib.packet.ethernet import ethernet

# Relabeling related
from controller_constants import switches_level_method, hosts_level_method, const_host_lvl, const_switch_lvl, max_sec_lvl, min_sec_lvl, delta_j, RELABELING_PERIOD
from controller_constants import SWITCHES, HOSTS, LINKS, NODES_LIST, SWITCH_MAC_TO_DPID
from controller_constants import MLSSwitch, MLSHost, MLSLink

import struct
import time
from collections import namedtuple
from random import shuffle, random, randint


log = core.getLogger()


class LLDPSender (object):
  """
  Sends out discovery packets
  """

  SendItem = namedtuple("LLDPSenderItem", ('dpid', 'port_num', 'packet'))

  # NOTE: This class keeps the packets to send in a flat list, which makes
  #      adding/removing them on switch join/leave or (especially) port
  #      status changes relatively expensive. Could easily be improved.

  # Maximum times to run the timer per second
  _sends_per_sec = 15

  def __init__(self, send_cycle_time, ttl=120):
    """
    Initialize an LLDP packet sender

    send_cycle_time is the time (in seconds) that this sender will take to
      send every discovery packet.  Thus, it should be the link timeout
      interval at most.

    ttl is the time (in seconds) for which a receiving LLDP agent should
      consider the rest of the data to be valid.  We don't use this, but
      other LLDP agents might.  Can't be 0 (this means revoke).
    """
    # Packets remaining to be sent in this cycle
    self._this_cycle = []

    # Packets we've already sent in this cycle
    self._next_cycle = []

    # Packets to send in a batch
    self._send_chunk_size = 1

    self._timer = None
    self._ttl = ttl
    self._send_cycle_time = send_cycle_time
    core.listen_to_dependencies(self)

  def _handle_openflow_PortStatus(self, event):
    """
    Track changes to switch ports
    """
    if event.added:
      self.add_port(event.dpid, event.port, event.ofp.desc.hw_addr)
    elif event.deleted:
      self.del_port(event.dpid, event.port)
    elif event.modified:
      if event.ofp.desc.config & of.OFPPC_PORT_DOWN == 0:
        # It's not down, so... try sending a discovery now
        self.add_port(event.dpid, event.port, event.ofp.desc.hw_addr, False)

  def _handle_openflow_ConnectionUp(self, event):
    self.del_switch(event.dpid, set_timer=False)

    ports = [(p.port_no, p.hw_addr) for p in event.ofp.ports]

    for port_num, port_addr in ports:
      self.add_port(event.dpid, port_num, port_addr, set_timer=False)

    self._set_timer()

  def _handle_openflow_ConnectionDown(self, event):
    self.del_switch(event.dpid)

  def del_switch(self, dpid, set_timer=True):
    self._this_cycle = [p for p in self._this_cycle if p.dpid != dpid]
    self._next_cycle = [p for p in self._next_cycle if p.dpid != dpid]
    if set_timer: self._set_timer()

  def del_port(self, dpid, port_num, set_timer=True):
    if port_num > of.OFPP_MAX: return
    self._this_cycle = [p for p in self._this_cycle
                        if p.dpid != dpid or p.port_num != port_num]
    self._next_cycle = [p for p in self._next_cycle
                        if p.dpid != dpid or p.port_num != port_num]
    if set_timer: self._set_timer()

  def add_port(self, dpid, port_num, port_addr, set_timer=True):
    if port_num > of.OFPP_MAX: return
    self.del_port(dpid, port_num, set_timer=False)
    packet = self.create_packet_out(dpid, port_num, port_addr)
    self._next_cycle.insert(0, LLDPSender.SendItem(dpid, port_num, packet))
    if set_timer: self._set_timer()
    core.openflow.sendToDPID(dpid, packet)  # Send one immediately

  def _set_timer(self):
    if self._timer: self._timer.cancel()
    self._timer = None
    num_packets = len(self._this_cycle) + len(self._next_cycle)

    if num_packets == 0: return

    self._send_chunk_size = 1  # One at a time
    interval = self._send_cycle_time / float(num_packets)
    if interval < 1.0 / self._sends_per_sec:
      # Would require too many sends per sec -- send more than one at once
      interval = 1.0 / self._sends_per_sec
      chunk = float(num_packets) / self._send_cycle_time / self._sends_per_sec
      self._send_chunk_size = chunk

    self._timer = Timer(interval,
                        self._timer_handler, recurring=True)

  def _timer_handler(self):
    """
    Called by a timer to actually send packets.

    Picks the first packet off this cycle's list, sends it, and then puts
    it on the next-cycle list.  When this cycle's list is empty, starts
    the next cycle.
    """
    num = int(self._send_chunk_size)
    fpart = self._send_chunk_size - num
    if random() < fpart: num += 1

    for _ in range(num):
      if len(self._this_cycle) == 0:
        self._this_cycle = self._next_cycle
        self._next_cycle = []
        # shuffle(self._this_cycle)
      item = self._this_cycle.pop(0)
      self._next_cycle.append(item)
      core.openflow.sendToDPID(item.dpid, item.packet)

  def create_packet_out(self, dpid, port_num, port_addr):
    """
    Create an ofp_packet_out containing a discovery packet
    """
    eth = self._create_discovery_packet(dpid, port_num, port_addr, self._ttl)
    po = of.ofp_packet_out(action=of.ofp_action_output(port=port_num))
    po.data = eth.pack()
    return po.pack()

  @staticmethod
  def _create_discovery_packet(dpid, port_num, port_addr, ttl):
    """
    Build discovery packet
    """

    chassis_id = pkt.chassis_id(subtype=pkt.chassis_id.SUB_LOCAL)
    chassis_id.id = ('dpid:' + hex(int(dpid))[2:]).encode()
    # Maybe this should be a MAC.  But a MAC of what?  Local port, maybe?

    port_id = pkt.port_id(subtype=pkt.port_id.SUB_PORT, id=str(port_num))

    ttl = pkt.ttl(ttl=ttl)

    sysdesc = pkt.system_description()
    sysdesc.payload = ('dpid:' + hex(int(dpid))[2:]).encode()

    discovery_packet = pkt.lldp()
    discovery_packet.tlvs.append(chassis_id)
    discovery_packet.tlvs.append(port_id)
    discovery_packet.tlvs.append(ttl)
    discovery_packet.tlvs.append(sysdesc)
    discovery_packet.tlvs.append(pkt.end_tlv())

    eth = pkt.ethernet(type=pkt.ethernet.LLDP_TYPE)
    eth.src = port_addr
    eth.dst = pkt.ETHERNET.NDP_MULTICAST
    eth.payload = discovery_packet

    return eth


class LinkEvent (Event):
  """
  Link up/down event
  """

  def __init__(self, add, link, event=None):
    self.link = link
    self.added = add
    self.removed = not add
    self.event = event  # PacketIn which caused this, if any

  def port_for_dpid(self, dpid):
    if self.link.dpid1 == dpid:
      return self.link.port1
    if self.link.dpid2 == dpid:
      return self.link.port2
    return None


class Link (namedtuple("LinkBase", ("dpid1", "port1", "dpid2", "port2"))):
  @property
  def uni(self):
    """
    Returns a "unidirectional" version of this link

    The unidirectional versions of symmetric keys will be equal
    """
    pairs = list(self.end)
    pairs.sort()
    return Link(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])

  @property
  def flipped(self):
    pairs = self.end
    return Link(pairs[1][0], pairs[1][1], pairs[0][0], pairs[0][1])

  @property
  def end(self):
    return ((self[0], self[1]), (self[2], self[3]))

  def __str__(self):
    return "%s.%s -> %s.%s" % (dpid_to_str(self[0]), self[1],
                               dpid_to_str(self[2]), self[3])

  def __repr__(self):
    return "Link(dpid1=%s,port1=%s, dpid2=%s,port2=%s)" % (self.dpid1,
        self.port1, self.dpid2, self.port2)


class Discovery (EventMixin):
  """
  Component that attempts to discover network toplogy.

  Sends out specially-crafted LLDP packets, and monitors their arrival.
  """

  _flow_priority = 65000     # Priority of LLDP-catching flow (if any)
  _link_timeout = 10         # How long until we consider a link dead
  _timeout_check_period = 5  # How often to check for timeouts

  _eventMixin_events = set([
    LinkEvent,
  ])

  _core_name = "openflow_discovery"  # we want to be core.openflow_discovery

  Link = Link

  def __init__(self, install_flow=True, explicit_drop=True,
                link_timeout=None, eat_early_packets=False):
    self._eat_early_packets = eat_early_packets
    self._explicit_drop = explicit_drop
    self._install_flow = install_flow
    if link_timeout: self._link_timeout = link_timeout

    # Only tracks switch->switch links by design
    self.adjacency = {}  # From Link to time.time() stamp
    self._sender = LLDPSender(self.send_cycle_time)

    # Should contain switch dict, host dict, links dict and nodesList
    self.network_graph = {}
    # DPID to obj
    self.network_graph[SWITCHES] = {}
    # MAC to obj
    self.network_graph[HOSTS] = {}
    # (source_id,dest_id) to obj (Unidirectional keys)
    self.network_graph[LINKS] = {}
    # Should be a list so populate later
    self.network_graph[NODES_LIST] = None
    # Switch MAC to DPID (all ports on switch map to same DPID)
    self.network_graph[SWITCH_MAC_TO_DPID]={}

    # Listen with a high priority (mostly so we get PacketIns early)
    core.listen_to_dependencies(self,
        listen_args={'openflow': {'priority': 0xffffffff}})

    Timer(self._timeout_check_period, self._expire_links, recurring=True)

  @property
  def send_cycle_time(self):
    return self._link_timeout / 2.0

  @property
  def get_network_graph(self):
    return self.network_graph

  def install_flow(self, con_or_dpid, priority=None):
    if priority is None:
      priority = self._flow_priority
    if isinstance(con_or_dpid, int):
      con = core.openflow.connections.get(con_or_dpid)
      if con is None:
        log.warn("Can't install flow for %s", dpid_to_str(con_or_dpid))
        return False
    else:
      con = con_or_dpid

    match = of.ofp_match(dl_type=pkt.ethernet.LLDP_TYPE,
                          dl_dst=pkt.ETHERNET.NDP_MULTICAST)
    msg = of.ofp_flow_mod()
    msg.priority = priority
    msg.match = match
    msg.actions.append(of.ofp_action_output(port=of.OFPP_CONTROLLER))
    con.send(msg)
    return True

  def _handle_openflow_ConnectionUp(self, event):
    if self._install_flow:

      # Make sure we get appropriate traffic
      log.debug("Installing flow for %s", dpid_to_str(event.dpid))
      # print(event.connection.features)
      self.install_flow(event.connection)

      switch_key = dpid_to_str(event.dpid)

      # Assign switch level (Irrespective of topo - Better?)
      # TODO - Can use attestation hook etc later
      if switches_level_method == 1:
        switch_level = const_switch_lvl
      elif switches_level_method == 2:
        switch_level = randint(min_sec_lvl, max_sec_lvl)

      port_info = {}
      for port in event.ofp.ports:
        port_info[port.port_no] = port.hw_addr.toStr()

      switch_object = MLSSwitch(switch_key, level=switch_level, port_info=port_info)
      # Update network info appropriately
      self.network_graph[SWITCHES][switch_key] = switch_object
      log.debug("Switch  %s assigned level %s",
                dpid_to_str(event.dpid), str(switch_level))

  def _handle_openflow_ConnectionDown(self, event):
    # Delete all links on this switch
    self._delete_links([link for link in self.adjacency
                        if link.dpid1 == event.dpid
                        or link.dpid2 == event.dpid])
    # Remove switch info
    # TODO - Delete links when switch down
    self.network_graph[SWITCHES].pop(dpid_to_str(event.dpid))

  def _expire_links(self):
    """
    Remove apparently dead links
    """
    now = time.time()

    expired = [link for link, timestamp in self.adjacency.items()
               if timestamp + self._link_timeout < now]
    if expired:
      for link in expired:
        log.info('link timeout: %s', link)

      self._delete_links(expired)

  def _handle_openflow_PacketIn(self, event):
    """
    Receive and process LLDP packets
    """

    packet = event.parsed

    if (packet.effective_ethertype != pkt.ethernet.LLDP_TYPE
        or packet.dst != pkt.ETHERNET.NDP_MULTICAST):
      if not self._eat_early_packets: return
      if not event.connection.connect_time: return
      enable_time = time.time() - self.send_cycle_time - 1
      if event.connection.connect_time > enable_time:
        return EventHalt
      return

    if self._explicit_drop:
      if event.ofp.buffer_id is not None:
        log.debug("Dropping LLDP packet %i", event.ofp.buffer_id)
        msg = of.ofp_packet_out()
        msg.buffer_id = event.ofp.buffer_id
        msg.in_port = event.port
        event.connection.send(msg)

    lldph = packet.find(pkt.lldp)
    if lldph is None or not lldph.parsed:
      log.error("LLDP packet could not be parsed")
      return EventHalt
    if len(lldph.tlvs) < 3:
      log.error("LLDP packet without required three TLVs")
      return EventHalt
    if lldph.tlvs[0].tlv_type != pkt.lldp.CHASSIS_ID_TLV:
      log.error("LLDP packet TLV 1 not CHASSIS_ID")
      return EventHalt
    if lldph.tlvs[1].tlv_type != pkt.lldp.PORT_ID_TLV:
      log.error("LLDP packet TLV 2 not PORT_ID")
      return EventHalt
    if lldph.tlvs[2].tlv_type != pkt.lldp.TTL_TLV:
      log.error("LLDP packet TLV 3 not TTL")
      return EventHalt

    def lookInSysDesc():
      r = None
      for t in lldph.tlvs[3:]:
        if t.tlv_type == pkt.lldp.SYSTEM_DESC_TLV:
          # This is our favored way...
          for line in t.payload.decode().split('\n'):
            if line.startswith('dpid:'):
              try:
                return int(line[5:], 16)
              except:
                pass
          if len(t.payload) == 8:
            # Maybe it's a FlowVisor LLDP...
            # Do these still exist?
            try:
              return struct.unpack("!Q", t.payload)[0]
            except:
              pass
          return None

    originatorDPID = lookInSysDesc()

    if originatorDPID == None:
      # We'll look in the CHASSIS ID
      if lldph.tlvs[0].subtype == pkt.chassis_id.SUB_LOCAL:
        if lldph.tlvs[0].id.startswith(b'dpid:'):
          # This is how NOX does it at the time of writing
          try:
            originatorDPID = int(lldph.tlvs[0].id[5:], 16)
          except:
            pass
      if originatorDPID == None:
        if lldph.tlvs[0].subtype == pkt.chassis_id.SUB_MAC:
          # Last ditch effort -- we'll hope the DPID was small enough
          # to fit into an ethernet address
          if len(lldph.tlvs[0].id) == 6:
            try:
              s = lldph.tlvs[0].id
              originatorDPID = struct.unpack("!Q", '\x00\x00' + s)[0]
            except:
              pass

    if originatorDPID == None:
      log.warning("Couldn't find a DPID in the LLDP packet")
      return EventHalt

    if originatorDPID not in core.openflow.connections:
      log.info('Received LLDP packet from unknown switch')
      return EventHalt

    # Get port number from port TLV
    if lldph.tlvs[1].subtype != pkt.port_id.SUB_PORT:
      log.warning("Thought we found a DPID, but packet didn't have a port")
      return EventHalt
    originatorPort = None
    if lldph.tlvs[1].id.isdigit():
      # We expect it to be a decimal value
      originatorPort = int(lldph.tlvs[1].id)
    elif len(lldph.tlvs[1].id) == 2:
      # Maybe it's a 16 bit port number...
      try:
        originatorPort = struct.unpack("!H", lldph.tlvs[1].id)[0]
      except:
        pass

    if originatorPort is None:
      log.warning("Thought we found a DPID, but port number didn't " +
                  "make sense")
      return EventHalt

    if (event.dpid, event.port) == (originatorDPID, originatorPort):
      log.warning("Port received its own LLDP packet; ignoring")
      return EventHalt

    link = Discovery.Link(originatorDPID, originatorPort, event.dpid,
                          event.port)

    if link not in self.adjacency:
      self.adjacency[link] = time.time()
      log.info('link detected: %s', link)    
      self.raiseEventNoErrors(LinkEvent, True, link, event)
    else:
      # Just update timestamp
      self.adjacency[link] = time.time()

    return EventHalt  # Probably nobody else needs this event

  def _delete_links(self, links):
    for link in links:
      self.raiseEventNoErrors(LinkEvent, False, link)  
    for link in links:  
      self.adjacency.pop(link, None)




  def is_edge_port(self, dpid, port):
    """
    Return True if given port does not connect to another switch
    """
    for link in self.adjacency:
      if link.dpid1 == dpid and link.port1 == port:
        return False
      if link.dpid2 == dpid and link.port2 == port:
        return False
    return True

  # Replace this with an Event later maybe(?) - More Robust Host discovery
  def set_host_info(self, mac_to_connection_info_tuple):
    """
      Used to update network graph with host info
      mac_to_connection_info_tuple dict of mac address(string) to (dpid,port) of switch it is connected to
    """
    # Create host objects (generate level). Again some mechanism/attestation
    for mac_addr in mac_to_connection_info_tuple:
      sw_dpid, sw_port = mac_to_connection_info_tuple[mac_addr]
      sw_object=self.network_graph[SWITCHES][sw_dpid]
      sw_level = sw_object.level
      # TODO - Add check here to ensure mac is a host mac
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
      host_obj=MLSHost(mac_addr, host_lvl)
      self.network_graph[HOSTS][mac_addr] = host_obj
      # Add the host<->switch link info 
      self.network_graph[LINKS][(mac_addr,sw_dpid)] = MLSLink(host_obj, sw_object,dest_port=sw_port)
      self.network_graph[LINKS][(sw_dpid,mac_addr)] = MLSLink(sw_object, host_obj,source_port=sw_port)

    
    # Now construct link matrix 
    for link in self.adjacency:
      source_dpid = dpid_to_str(link.dpid1)
      source_obj=self.network_graph[SWITCHES][source_dpid]
      source_port=link.port1
      dest_dpid = dpid_to_str(link.dpid2)
      dest_obj=self.network_graph[SWITCHES][dest_dpid]
      dest_port=link.port2

      # if (source_dpid1,dest_dpid) not in self.network_graph[LINKS].keys():
      self.network_graph[LINKS][source_dpid,dest_dpid] = MLSLink(source_obj, dest_obj,source_port=source_port,dest_port=dest_port)
      self.network_graph[LINKS][(dest_dpid,source_dpid)] = MLSLink(dest_obj, source_obj,source_port=dest_port,dest_port=source_port)

    log.info("# Switches: %s",str(len(self.network_graph[SWITCHES].keys())))
    log.info("# (Bi directional) links found: %s", str(len(self.network_graph[LINKS].keys())/2))

    # Populate switch mac to dpid dict
    for sw_dpid in self.network_graph[SWITCHES].keys():
      for port_no in  self.network_graph[SWITCHES][sw_dpid].port_info:
        mac_addr=self.network_graph[SWITCHES][sw_dpid].port_info[port_no]
        self.network_graph[SWITCH_MAC_TO_DPID][mac_addr]=sw_dpid

    log.info("# Switch MAC addresses learned: %s",str(len(self.network_graph[SWITCH_MAC_TO_DPID].keys())))
     





def launch(no_flow=False, explicit_drop=True, link_timeout=None,
            eat_early_packets=False):
  explicit_drop=str_to_bool(explicit_drop)
  eat_early_packets=str_to_bool(eat_early_packets)
  install_flow=not str_to_bool(no_flow)
  if link_timeout: link_timeout=int(link_timeout)

  core.registerNew(Discovery, explicit_drop=explicit_drop,
                   install_flow=install_flow, link_timeout=link_timeout,
                   eat_early_packets=eat_early_packets)
