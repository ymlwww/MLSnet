#!/usr/bin/env python3.7


from random import choice, shuffle, randint

import sys
import socket
import struct
from time import sleep
import numpy as np

from copy import deepcopy
from multiprocessing import Pool
from threading import Thread
from mininet.node import Node
from topo import MeshTopo
import time
# from scapy.all import *


from common import burst_lvl, burst_ratio
from common import min_sec_lvl, max_sec_lvl, optimization_problem, DEBUG
from common import min_flow_demand, max_flow_demand, const_flow_lvl, flows_level_method
from common import LINK_FAILURE_DURATION, LINK_FAILURE_FLAG, LINK_FAILURE_PERC, LINK_FAILURE_PERIOD
from common import NUM_FLOWS, FLOW_GENERATION_PERIOD, TOTAL_EXPERIMENTATION_EPOCHS, MIN_FLOW_DURATION

# Size in bytes
PACKET_PAYLOAD_SIZE = 32
network = None
topo_info = None
host_to_ip = {}
host_to_mac = {}
number_of_links_to_fail = None
current_failed_links = set()

### Classes ###


class Flow():
    """A flow of packets originating from a specified source toward a specified destination, with given level and demand (in bps)."""

    def __init__(self, key, source_label, dest_label, level, demand, duration):
        self.key = key
        self.source = source_label
        self.dest = dest_label
        self.level = level
        self.demand = demand
        self.retries = 0
        self.duration = duration
        self.wait_time_remaining = -1


def fail_links():
    """
        Fails links

    """
    global network
    global number_of_links_to_fail
    global current_failed_links
    current_failed_links.clear()

    # Bi directional links. Only things which are passed to librelabelling need to have uni diretional links (Controller network graph links ds and failed links ds)
    for link_key in topo_info.getlinks:
        # Pick a random sw-sw link
        if link_key not in current_failed_links and len(current_failed_links)<number_of_links_to_fail:
            if link_key[0] not in host_to_ip and link_key[1] not in host_to_ip:
                current_failed_links.add(link_key)
                # Fail the link now
                network.configLinkStatus(link_key[0], link_key[1], "down")

    print("Failed links - ", current_failed_links)


def restore_links():
    global network
    global current_failed_links

    # Bi directional links. Only things which are passed to librelabelling need to have uni diretional links (Controller network graph links ds and failed links ds)
    for link_key in current_failed_links:
        # Restore the link
        network.configLinkStatus(link_key[0], link_key[1], "up")

    print("Restored links - ", current_failed_links)
    current_failed_links.clear()


def run_experiment(net, topo):
    global network, topo_info
    global host_to_ip, host_to_mac
    global epoch_no
    global f
    global number_of_links_to_fail
    f = open("flow_out_new.txt", "w")
    targets_per_epoch = []
    network = net
    topo_info = topo
    for host_obj in network.hosts:
        host_to_ip[host_obj.name] = host_obj.IP()
        host_to_mac[host_obj.name] = host_obj.MAC()

    if LINK_FAILURE_FLAG:
        number_of_links_to_fail = int(len(network.links)*LINK_FAILURE_PERC)
    epoch_no = 1

    while epoch_no <= TOTAL_EXPERIMENTATION_EPOCHS:
        flows = gen_flows(NUM_FLOWS, topo._hosts)

        if LINK_FAILURE_FLAG and epoch_no % LINK_FAILURE_PERIOD == 0 and epoch_no != TOTAL_EXPERIMENTATION_EPOCHS:
            # Fail some links for this epoch (Can make this generic i.e link failure duration does not have to be epoch duration later)
            fail_links()

        elif LINK_FAILURE_FLAG and epoch_no % LINK_FAILURE_PERIOD == LINK_FAILURE_DURATION:
            restore_links()

        targets = generate_packets(flows)
        targets_per_epoch.append(targets)
        # Add some sleep here
        sleep(FLOW_GENERATION_PERIOD)

        epoch_no = epoch_no+1

    # Wait for the last bunch of flows
    sleep(MIN_FLOW_DURATION)

    get_flow_stats(targets_per_epoch)


def gen_flows(num_flows, hosts_dict, burst_event=False):
    """Generates a specified number of flows (source,dest) given a set of hosts.
       Level can be generated and passed in a custom packet but we instead assign levels in the controller.
    """

    # generate durations with pareto dist
    xm1 = MIN_FLOW_DURATION
    a1 = 0.8
    # durs1 = (np.random.pareto(a1, int(num_flows/2.0)) + 1) * xm1
    # durs = (np.random.pareto(a1, num_flows) + 1) + xm1
    durs = [MIN_FLOW_DURATION for _ in range(num_flows)]
    print("Durations (seconds):", durs)
    host_identifiers = list(hosts_dict.keys())

    shuffle(durs)

    flow_lvls = [lvl for lvl in range(min_sec_lvl, max_sec_lvl + 1)
                 for _ in range(num_flows // (max_sec_lvl - min_sec_lvl + 1))]  # control distribution of flow values

    if burst_event:
        flow_skew_ratio = 0.95
        num_non_skewed_flows = int(flow_skew_ratio*num_flows)
        num_skewed_flows = num_flows - \
            (num_non_skewed_flows-(num_non_skewed_flows % 3))
        flow_lvls = [lvl for lvl in range(min_sec_lvl, max_sec_lvl) for _ in range(
            (num_flows-num_skewed_flows)//3)]
        flow_lvls += [max_sec_lvl for _ in range(num_skewed_flows)]
        # print(num_skewed_flows)
        # print(flow_lvls)
    shuffle(flow_lvls)  # add some more randomness

    flows = set()
    for i in range(num_flows):
        if burst_event and (
                len(flows) <= (num_flows * burst_ratio)):
            fsrc = choice(host_identifiers)
            fsrc = hosts_dict[fsrc]
            retries = 100
            while (fsrc.level != burst_lvl) and (retries > 0):
                # find a suitable source for this flow in the burst
                fsrc = choice(host_identifiers)
                fsrc = hosts_dict[fsrc]
                retries -= 1
            fdst = choice(host_identifiers)
            fdst = hosts_dict[fdst]
            # warmup to steady state #flows then steady state switch levels
            # dont burst at epoch 0, only after warmup, and burst every 10th epoch (not 0 indexed)
            # while (fdst is fsrc) or (eq_src_dst_flag and (fdst.level != fsrc.level)) or (min(fsrc.level, fdst.level) != (burst_lvl if (time_epoch_idx/10) % 2 == 0 else 3)):
            # while (fdst is fsrc) or (eq_src_dst_flag and (fdst.level != fsrc.level)) or (min(fsrc.level, fdst.level) != burst_lvl):
            while (fdst is fsrc) or ((optimization_problem == 1) and (fdst.level < fsrc.level)) or (
                    (optimization_problem == 2) and (fdst.level != fsrc.level)):
                # find distinct src/dest that are of appropriate level
                # maybe do this in else case too to add more randomness
                # fsrc = choice(hosts)  # NOTE 7/22/2020: controlling flow levels now so dont need this
                # for OPT1 the if check above finds a dest that is >= so the flow level will be the source level (maybe also dest) as intended
                # for OPT2 it finds a dest of equal level so flow will be equal to src and dest as intended
                fdst = choice(host_identifiers)
                fdst = hosts_dict[fdst]
            # f = Flow(i, fsrc, fdst, burst_lvl if (time_epoch_idx/10) % 2 == 0 else 3, randint(
            #     min_flow_demand, max_flow_demand), durs[i-key_start])  # level 3 burst on odds
            f = Flow(i, fsrc, fdst, burst_lvl, randint(
                min_flow_demand, max_flow_demand), durs[i])
        else:
            fsrc = choice(host_identifiers)
            fsrc = hosts_dict[fsrc]
            retries = 100
            while (fsrc.level != flow_lvls[i]) and (retries > 0):
                # find a suitable source for this level
                fsrc = choice(host_identifiers)
                fsrc = hosts_dict[fsrc]
                retries -= 1

            fdst = choice(host_identifiers)
            fdst = hosts_dict[fdst]

            while (fdst is fsrc) or ((optimization_problem == 1) and (fdst.level < fsrc.level)) or (
                    (optimization_problem == 2) and (fdst.level != fsrc.level)):
                # while (fdst is fsrc) or (balance_flows_flag and (fdst.level < fsrc.level)) or (eq_src_dst_flag and (fdst.level != fsrc.level)):
                # while (fdst is fsrc) or (balance_flows_flag and (fdst.level < fsrc.level)) or (eq_src_dst_flag and (fdst.level != fsrc.level)) or (burst_flag and (min(fsrc.level, fdst.level) == burst_lvl)): # ignore burst_lvl
                # fsrc = choice(hosts) # NOTE 7/22/2020: controlling flow levels now so dont need this
                # for OPT1 the if check above finds a dest that is >= so the flow level will be the source level (maybe also dest) as intended
                # for OPT2 it finds a dest of equal level so flow will be equal to src and dest as intended
                fdst = choice(host_identifiers)
                fdst = hosts_dict[fdst]
            if flows_level_method == 1:  # fixed
                f = Flow(i, fsrc.label, fdst.label, const_flow_lvl, randint(
                    min_flow_demand, max_flow_demand), durs[i])
            elif flows_level_method == 2:  # random
                f = Flow(i, fsrc.label, fdst.label, randint(min_sec_lvl, max_sec_lvl),
                         randint(min_flow_demand, max_flow_demand), durs[i])
            if flows_level_method == 3:  # == source
                f = Flow(i, fsrc.label, fdst.label, fsrc.level, randint(
                    min_flow_demand, max_flow_demand), durs[i])
            elif flows_level_method == 4:  # <= dest
                f = Flow(i, fsrc.label, fdst.label, randint(min_sec_lvl, fdst.level),
                         randint(min_flow_demand, max_flow_demand), durs[i])
            elif flows_level_method == 5:  # == min(src,dst) or <= min(src,dst)
                f = Flow(i, fsrc.label, fdst.label, min(fsrc.level, fdst.level),
                         randint(min_flow_demand, max_flow_demand), durs[i])
                # f = Flow(i, fsrc, fdst, randint(min_sec_lvl, min(fsrc.level, fdst.level)), randint(
                #     min_flow_demand, max_flow_demand), durs[i-key_start])
        flows.add(f)

    return flows


def generate_packets(flows):

    target_hosts = set()
    flow_ct = 1

    for flow in flows:
        target_hosts.add(flow.dest)
        dst_ip = host_to_ip[flow.dest]
        packets_per_second = 100

        # Nested loop may cause unnecessary overhead- Have extra dictionary?
        for host_obj in network.hosts:

            if (str(host_obj.name) == str(flow.source)):
                # print("Packets per second",str(packets_per_second),str(flow.demand),str(PACKET_PAYLOAD_SIZE))
                interval = float(1/packets_per_second)
                cmd = ('ping %s -c %s -i %s >> epoch%s_flow%s.txt &' %
                       (dst_ip, int(flow.duration*packets_per_second), interval, epoch_no, flow_ct))
                print("Flow %s -> %s for %s" % (flow.source, flow.dest,
                      str(int(flow.duration*packets_per_second))))
                print("     %s -> %s" %
                      (host_to_mac[flow.source], host_to_mac[flow.dest]))
                host_obj.cmd(cmd, printPid=False)

        flow_ct = flow_ct + 1

    return target_hosts



def get_flow_stats(targets_per_epoch):
    # Gettings stats on packet loss
    for epoch in range(TOTAL_EXPERIMENTATION_EPOCHS):
        f.write("EPOCH NUMBER: %s\n" % (epoch))
        target_hosts = targets_per_epoch[epoch]

        for target in target_hosts:
            for host_obj in network.hosts:
                if (str(host_obj.name) == target):
                    for ct in range(1, NUM_FLOWS + 1):
                        result = host_obj.cmd("cat epoch%s_flow%s.txt" %
                                              (epoch+1, ct))
                        if (result.split(' ')[0] != 'cat:'):
                            f.write("Flow[%s] Ping Ouput \n" %
                                    (ct))
                            f.write(result)
                            # print(result)
                            host_obj.cmd("rm epoch%s_flow%s.txt" % (epoch+1, ct))
