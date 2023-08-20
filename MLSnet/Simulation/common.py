#!/usr/bin/env python3.7
#
# common.py
#
# Description   : Common classes and methods.
# Created by    :
# Date          : November 2019
# Last Modified : July 2020


### Imports ###
from random import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

### Globals ###
DEBUG = False
measure_agility= False
show_sec_metric_flag =False
optimization_problem = 1
delta_j = 2
delta_c_j = 50e6
B = 5
min_sec_lvl = 1
max_sec_lvl = 4
num_levels = max_sec_lvl - min_sec_lvl + 1

# relabeling frequency (should be >= reboot time; otherwise we end up relabeling again before a reboot finishes and the flows even get routed)
num_warmup_epochs = 100 # must be exactly at the relabeling_period or will be misaligned with solver since the num_warmup_epochs checks are kinda diff
relabeling_period = 100
reboot_time = 10

weight_of_rebooting_switch = 10e6

max_flow_retries = 0
flow_weight_power = 2

burst_flag = False  # simulate bursts
burst_start_idx = 1220
burst_duration = 700 # Indefinite
burst_period = 1220
# not 0 indexed; set high to ignore; should be >warmup
burst_ratio = 0.9  # ratio of flows arriving that have burst_lvl
burst_lvl = 4  # level of burst flows

link_fails_flag = False  # simulate link fails
link_fails_start_idx = 1220
link_fail_duration = 700  # simulate fail for n secs; <= link_fail_period
link_fail_period = 1220
link_fail_perc = 0.50

host_migration_flag = False  # simulate host (subnet) migrations
host_migration_start_idx = 1220
host_migration_period = 1220
host_migration_perc = 0.50  # set <= 0.5 since we simulate by swapping subnet locations

# balance_flows_flag = False  # set True and hosts_level_method=5/6 to balance # OPT1
# eq_src_dst_flag = True  # auto checks if doing OPT2
switches_level_method = 2  # 1-const, 2-random, 3-other
hosts_level_method = 6  # 1-const, 2-random, 3-other, 5-==sw_lvl, 6-subnet
flows_level_method = 5  # 1-const, 2-random, 5-==min(src,dst)
const_switch_lvl = max_sec_lvl
const_host_lvl = const_switch_lvl
const_flow_lvl = const_switch_lvl
hosts_per_wan_switch = 5
hosts_per_mesh_switch = 5

min_flow_demand = 50e6  # bps
max_flow_demand = 50e6
agg_to_core_link_cap = 10e9
edge_to_agg_link_cap = 10e9
host_to_edge_link_cap = 1e9
mesh_switch_link_cap = 10e9
default_link_cap = 1e9
switch_cap = 100e9


### Classes ###
class Switch():
    """A switch node in the network."""

    def __init__(self, key, level, capacity=float("inf")):
        self.key = key
        self.level = level
        self.capacity = capacity
        self.wait_time_remaining = -1


class Link():
    """A link between sources, switches, or destinations in the network."""

    def __init__(self, k, l, capacity=float("inf")):
        self.k = k
        self.l = l
        self.capacity = capacity


class Host():
    """A host node in the network."""

    def __init__(self, key, level):
        self.key = key
        self.level = level


class FatTreeHost(Host):
    """A host node in a fat-tree network."""
    pass


class Source(Host):
    """A source node in the network."""
    pass


class Dest(Host):
    """A destination node in the network."""
    pass


class Flow():
    """A flow of packets originating from a specified source toward a specified destination, with given level and demand (in bps)."""

    def __init__(self, key, source, dest, level, demand, duration):
        self.key = key
        self.source = source
        self.dest = dest
        self.level = level
        self.demand = demand
        self.retries = 0
        self.duration = duration
        self.wait_time_remaining = -1


### Functions ###
def extractLinkMatrix(g, nodesList):
    """Extracts matrix representation for experimentation (lp and heuristic algorithm)."""
    lm_tmp = {r.key: {c.key: str(0) for c in nodesList} for r in nodesList}
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS'].keys():
                lm_tmp[k.key][l.key] = str(1)
            else:
                lm_tmp[k.key][l.key] = str(0)
    return lm_tmp


def get_num_route_downs(g, flow, flow_path):
    """Computes the number of route downs for a flow."""
    curr_node = flow_path[0]
    num_route_downs = 0
    for i in range(1, len(flow_path)):
        if (curr_node, flow_path[i]) in g['LINKS'].keys():
            if (flow.level > g['LINKS'][(curr_node, flow_path[i])].k.level):
                # dest must be ignored because a flow cannot be routed down to a dest
                num_route_downs += 1
        else:
            print("why is this not in links")
            exit(1)
        curr_node = flow_path[i]
    return num_route_downs


def get_info_type(node, switches, hosts, fstr=False):
    """Returns the correct dictionary for indexing purposes in the solver/heuristic."""
    if node in switches:  # overload use for dicts and lists
        if fstr:
            return 'SWITCHES', switches
        else:
            return switches
    elif node in hosts:
        if fstr:
            return 'HOSTS', hosts
        else:
            return hosts
    else:
        print("node (%s) is bad in get_info_type" % node)
        print("switches: ", switches)
        print("hosts: ", hosts)
        exit(1)
    return


def gen_random_topo(num_switches, num_flows_for_given_num_sw):
    """Generates a topology with random links and levels for nodes."""

    # TODO: for random networks, dont let any node be src/dest, we should delegate specific nodes as hosts and switches so we are not trying to upgrade hosts
    dests = [Dest("d" + str(i), randint(1, 4))
             for i in range(num_dests_r)]
    sources = [Source("s" + str(i), randint(1, dests[i].level))
               for i in range(num_sources)]
    switches = [Switch(str(i), randint(1, 4), switch_cap)
                for i in range(num_switches)]
    nodesList = sources + switches + dests  # sorted

    links = dict()

    # Source-Switch links
    for so in sources:
        for sw in switches:
            if randint(1, 100) < randint(1, 15):
                # Source to switch only
                links[(so.key, sw.key)] = Link(so, sw, default_link_cap)
                links[(sw.key, so.key)] = Link(sw, so, default_link_cap)

    # Switch-Switch links
    for s1 in switches:
        for s2 in switches:
            if randint(1, 100) < randint(1, 20) and s1 is not s2:
                links[(s1.key, s2.key)] = Link(s1, s2, default_link_cap)
                # add bidirectional link
                links[(s2.key, s1.key)] = Link(s2, s1, default_link_cap)

    # Switch-Dests links
    for so in switches:
        for de in dests:
            if randint(1, 100) < randint(1, 15):
                links[(so.key, de.key)] = Link(so, de, default_link_cap)
                links[(de.key, so.key)] = Link(de, so, default_link_cap)

    graph = {
        "SWITCHES": switches,
        "LINKS": links,
        "SOURCES": sources,
        "DESTS": dests,
    }

    linkMatrix = extractLinkMatrix(graph, nodesList)
    if (DEBUG):
        for node in linkMatrix.keys():
            print("Links from: ", node, " -> { ", end=" ")
            for nodeB in nodesList:
                if linkMatrix[node][nodeB.key] == str(1):
                    print(nodeB.key, end=" ")
            print("}")

    show_graph(graph, nodesList)

    return graph, nodesList


def gen_fat_tree_topo(k=4):
    """Generates a fat-tree topology with specified port density (k) per switch.
    Citation: Mohammad Al-Fares, Alexander Loukissas, and Amin Vahdat. 2008. A scalable, commodity data center network architecture. SIGCOMM Comput. Commun. Rev. 38, 4 (October 2008), 63–74. DOI:https://doi.org/10.1145/1402946.1402967.
    Link: http://ccr.sigcomm.org/online/files/p63-alfares.pdf.
    """
    switches_dict = dict()
    hosts_dict = dict()
    links = dict()
    if DEBUG:
        print("Fat-Tree(k=%d, levels=3)" % k)
    num_core_sw = (k / 2) ** 2
    num_edge_sw = 2 * ((k / 2) ** 2)
    num_total_sw = 5 * ((k / 2) ** 2)
    num_agg_sw = num_edge_sw
    num_hosts = 2 * ((k / 2) ** 3)
    pod_width = k / 2
    stride_length = pod_width
    num_pods = k

    max_core_lvl_for_this_pod_switch = {core_start_idx: 0 for core_start_idx in range(
        0, int(pod_width) * int(num_pods), int(stride_length))}
    max_agg_lvl_for_pod = {pod_idx: 0 for pod_idx in range(int(num_pods))}
    for pod_idx in range(int(num_pods)):
        sw_start_id_in_pod = pod_idx * pod_width
        for pod_pos in range(int(pod_width)):
            sw_idx = sw_start_id_in_pod + pod_pos
            core_start_idx = pod_pos * stride_length
            # for each of k/2 ports on the switch; only depends on pos in pod
            for port in range(int(stride_length)):  # add uplink core switches
                if not (('c' + str(int(core_start_idx + port))) in switches_dict):
                    if switches_level_method == 1:  # fixed
                        switch_lvl = const_switch_lvl
                    else:  # cover method 2 and 3 here
                        switch_lvl = randint(min_sec_lvl, max_sec_lvl)
                    switches_dict['c' + str(int(core_start_idx + port))] = Switch('c' +
                                                                                  str(int(
                                                                                      core_start_idx + port)),
                                                                                  switch_lvl, switch_cap)
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
            for port in range(int(stride_length)):  # add the agg switches
                if DEBUG:
                    print("max_core_lvl_for_this_pod_switch[core_start_idx:%d]: %d" % (
                        core_start_idx, max_core_lvl_for_this_pod_switch[core_start_idx]))
                if not (('asw' + str(int(sw_idx))) in switches_dict):
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
                    switches_dict['asw' + str(int(sw_idx))] = Switch(
                        'asw' + str(int(sw_idx)), switch_lvl, switch_cap)
                    if switch_lvl > max_agg_lvl_for_pod[pod_idx]:
                        max_agg_lvl_for_pod[pod_idx] = switch_lvl
                else:
                    if DEBUG:
                        print("already added agg switch")

            for port in range(int(stride_length)):  # add links from agg to core
                if ('asw' + str(int(sw_idx)), 'c' + str(int(core_start_idx + port))) not in links.keys():
                    links[('asw' + str(int(sw_idx)), 'c' + str(int(core_start_idx + port)))] = Link(
                        switches_dict['asw' +
                                      str(int(sw_idx))], switches_dict['c' + str(int(core_start_idx + port))],
                        agg_to_core_link_cap)
                if ('c' + str(int(core_start_idx + port)), 'asw' + str(int(sw_idx))) not in links.keys():
                    links[('c' + str(int(core_start_idx + port)), 'asw' + str(int(sw_idx)))] = Link(switches_dict['c' +
                                                                                                                  str(
                                                                                                                      int(
                                                                                                                          core_start_idx + port))],
                                                                                                    switches_dict[
                                                                                                        'asw' + str(int(
                                                                                                            sw_idx))],
                                                                                                    agg_to_core_link_cap)

            for port in range(int(stride_length)):  # add the agg switches
                if DEBUG:
                    print("max_core_lvl_for_this_pod_switch[core_start_idx:%d]: %d" % (
                        core_start_idx, max_core_lvl_for_this_pod_switch[core_start_idx]))
                if not (('asw' + str(int(sw_start_id_in_pod + port))) in switches_dict):
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
                    switches_dict['asw' + str(int(sw_start_id_in_pod + port))] = Switch(
                        'asw' + str(int(sw_start_id_in_pod + port)), switch_lvl, switch_cap)
                    if switch_lvl > max_agg_lvl_for_pod[pod_idx]:
                        max_agg_lvl_for_pod[pod_idx] = switch_lvl
                else:
                    if DEBUG:
                        print("already added agg switch5")

            # for each edge switch
            for port in range(int(stride_length)):
                # add link to agg switch
                if DEBUG:
                    print("max_agg_lvl_for_pod[pod_idx:%d]: %d" % (
                        pod_idx, max_agg_lvl_for_pod[pod_idx]))
                if not (('esw' + str(int(sw_idx))) in switches_dict):
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
                    switches_dict['esw' + str(int(sw_idx))] = Switch(
                        'esw' + str(int(sw_idx)), switch_lvl, switch_cap)
                else:
                    if DEBUG:
                        print("already added edge switch")

                if ('esw' + str(int(sw_idx)), 'asw' + str(int(sw_start_id_in_pod + port))) not in links.keys():
                    links[('esw' + str(int(sw_idx)), 'asw' + str(int(sw_start_id_in_pod + port)))] = Link(
                        switches_dict['esw' +
                                      str(int(sw_idx))], switches_dict['asw' + str(int(sw_start_id_in_pod + port))],
                        edge_to_agg_link_cap)
                if ('asw' + str(int(sw_start_id_in_pod + port)), 'esw' + str(int(sw_idx))) not in links.keys():
                    links[('asw' + str(int(sw_start_id_in_pod + port)), 'esw' + str(int(sw_idx)))] = Link(
                        switches_dict['asw' +
                                      str(int(sw_start_id_in_pod + port))], switches_dict['esw' + str(int(sw_idx))],
                        edge_to_agg_link_cap)

            if DEBUG:
                print("===")

            subnet_lvl = randint(min_sec_lvl, max_sec_lvl)
            for port in range(int(stride_length)):
                # also add a link to hosts for each port
                if not (('h' + str(int((sw_idx * stride_length) + port))) in hosts_dict):
                    if hosts_level_method == 1:  # fixed
                        switch_lvl = const_host_lvl  # overload use of switch_lvl for hosts
                    elif hosts_level_method == 2:  # random
                        switch_lvl = randint(min_sec_lvl, max_sec_lvl)
                    elif hosts_level_method == 3:  # other1
                        switch_lvl = randint(max(
                            min_sec_lvl, switches_dict['esw' + str(int(sw_idx))].level - delta_j),
                            switches_dict['esw' + str(int(sw_idx))].level)
                    elif hosts_level_method == 4:  # other2
                        switch_lvl = min(1 + (pod_idx + pod_pos) % max_sec_lvl,
                                         switches_dict['esw' + str(int(sw_idx))].level)
                    elif hosts_level_method == 5:  # other3 (same as parent)
                        switch_lvl = switches_dict['esw' +
                                                   str(int(sw_idx))].level
                    elif hosts_level_method == 6:  # other4 (same in subnet)
                        switch_lvl = subnet_lvl
                    hosts_dict['h' + str(int((sw_idx * stride_length) + port))] = FatTreeHost(
                        'h' + str(int((sw_idx * stride_length) + port)), switch_lvl)
                else:
                    if DEBUG:
                        print("already added host")

                if ('esw' + str(int(sw_idx)), 'h' + str(int((sw_idx * stride_length) + port))) not in links.keys():
                    links[('esw' + str(int(sw_idx)), 'h' + str(int((sw_idx * stride_length) + port)))] = Link(
                        switches_dict['esw' +
                                      str(int(sw_idx))], hosts_dict['h' + str(int((sw_idx * stride_length) + port))],
                        host_to_edge_link_cap)
                if ('h' + str(int((sw_idx * stride_length) + port)), 'esw' + str(int(sw_idx))) not in links.keys():
                    links[('h' + str(int((sw_idx * stride_length) + port)), 'esw' + str(int(sw_idx)))] = Link(
                        hosts_dict['h' +
                                   str(int((sw_idx * stride_length) + port))], switches_dict['esw' + str(int(sw_idx))],
                        host_to_edge_link_cap)

    switches = sorted([s for s in switches_dict.values()],
                      key=lambda s: s.key[3:])

    hosts = list(hosts_dict.values())
    nodesList = switches + hosts
    graph = {
        "SWITCHES": switches,
        "LINKS": links,
        "HOSTS": hosts,
        "nodesList": nodesList
    }

    if DEBUG:
        print("hosts: ", list(hosts_dict.values()))
        print("nodesList: ", [n.key for n in nodesList])

    return graph, num_total_sw, num_hosts


def gen_wan_topo_A():
    """Generates the ASN WAN topology from the topo file."""
    WAN = nx.read_gml("Topo/ISP_WAN/A/AttNA.gml", label="id")
    switches_dict = dict()
    hosts_dict = dict()
    links_dict = dict()

    for sw_pos in range(WAN.number_of_nodes()):
        sw_label = "sw" + str(sw_pos)
        if switches_level_method == 1:  # fixed
            sw_level = const_switch_lvl
        elif switches_level_method == 2:  # random
            sw_level = randint(min_sec_lvl, max_sec_lvl)
        switches_dict[sw_label] = Switch(sw_label, sw_level, switch_cap)
        host_lvl = 0
        subnet_lvl = randint(min_sec_lvl, max_sec_lvl)
        for host_pos in range(hosts_per_wan_switch):
            host_label = 'h' + str((sw_pos * hosts_per_wan_switch + host_pos))
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
            hosts_dict[host_label] = Host(host_label, host_lvl)
            # Add switch-host link
            links_dict[(sw_label, host_label)] = Link(switches_dict[sw_label], hosts_dict[host_label],
                                                      host_to_edge_link_cap)
            # Add host-switch link
            links_dict[(host_label, sw_label)] = Link(hosts_dict[host_label], switches_dict[sw_label],
                                                      host_to_edge_link_cap)

    for edge in WAN.edges():
        sw_from_label = "sw" + str(edge[0])
        sw_dest_label = "sw" + str(edge[1])
        links_dict[(sw_from_label, sw_dest_label)] = Link(switches_dict[sw_from_label], switches_dict[sw_dest_label],
                                                          edge_to_agg_link_cap)
        # Add the bi link
        links_dict[(sw_dest_label, sw_from_label)] = Link(switches_dict[sw_dest_label], switches_dict[sw_from_label],
                                                          edge_to_agg_link_cap)

    num_total_sw = WAN.number_of_nodes()
    num_hosts = num_total_sw * hosts_per_wan_switch
    switches = sorted([s for s in switches_dict.values()],
                      key=lambda s: s.key[3:])

    hosts = list(hosts_dict.values())
    nodesList = switches + hosts
    graph = {
        "SWITCHES": switches,
        "LINKS": links_dict,
        "HOSTS": hosts,
        "nodesList": nodesList
    }
    if DEBUG:
        print("hosts: ", list(hosts_dict.values()))
        print("nodesList: ", [n.key for n in nodesList])
    return graph, num_total_sw, num_hosts


def flow_weight(level):
    """Computes the weight of a flow given the level."""
    if level <= max_sec_lvl:
        return level ** flow_weight_power
    else:
        print("Bad level in flow_weight(): %d." % level)
        exit(1)


def gen_flows(key_start, num_flows, hosts, time_epoch_idx):
    """Generates a specified number of flows (with starting key counter key_start) given a set of hosts."""

    # generate durations with pareto dist
    xm1 = 0.001
    a1 = 1.1
    # durs1 = (np.random.pareto(a1, int(num_flows/2.0)) + 1) * xm1
    durs = (np.random.pareto(a1, num_flows) + 1) * xm1
    # xm2 = 1
    # a2 = 1.1
    # durs2 = (np.random.pareto(a2, int(num_flows/2.0)) + 1) * xm2
    # durs = np.concatenate([durs1, durs2])

    shuffle(durs)

    flow_lvls = [lvl for lvl in range(min_sec_lvl, max_sec_lvl + 1)
                 for _ in range(num_flows // (max_sec_lvl - min_sec_lvl + 1))]  # control distribution of flow values

    if burst_flag :
        flow_skew_ratio =0.95
        num_non_skewed_flows=int(flow_skew_ratio*num_flows)
        num_skewed_flows = num_flows- (num_non_skewed_flows-(num_non_skewed_flows%3))
        flow_lvls = [lvl for lvl in range(min_sec_lvl, max_sec_lvl)   for _ in range((num_flows-num_skewed_flows)//3)]
        flow_lvls+= [max_sec_lvl for _ in range(num_skewed_flows)]
        # print(num_skewed_flows)
        # print(flow_lvls)
    shuffle(flow_lvls)  # add some more randomness

    if burst_flag and (time_epoch_idx >= burst_start_idx) and (time_epoch_idx % burst_period < burst_duration) and DEBUG:
        print("Burst epoch - ", time_epoch_idx)

    flows = set()
    for i in range(key_start, key_start + num_flows):
        if burst_flag and (time_epoch_idx >= burst_start_idx) and (time_epoch_idx % burst_period < burst_duration) and (
                len(flows) <= (num_flows * burst_ratio)):
            fsrc = choice(hosts)
            retries = 100
            while (fsrc.level != burst_lvl) and (retries > 0):
                # find a suitable source for this flow in the burst
                fsrc = choice(hosts)
                retries -= 1
            fdst = choice(hosts)
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
                fdst = choice(hosts)
            # f = Flow(i, fsrc, fdst, burst_lvl if (time_epoch_idx/10) % 2 == 0 else 3, randint(
            #     min_flow_demand, max_flow_demand), durs[i-key_start])  # level 3 burst on odds
            f = Flow(i, fsrc, fdst, burst_lvl, randint(
                min_flow_demand, max_flow_demand), durs[i - key_start])
        else:
            fsrc = choice(hosts)
            retries = 100
            while (fsrc.level != flow_lvls[i - key_start]) and (retries > 0):
                # find a suitable source for this level
                fsrc = choice(hosts)
                retries -= 1

            fdst = choice(hosts)
            while (fdst is fsrc) or ((optimization_problem == 1) and (fdst.level < fsrc.level)) or (
                    (optimization_problem == 2) and (fdst.level != fsrc.level)):
                # while (fdst is fsrc) or (balance_flows_flag and (fdst.level < fsrc.level)) or (eq_src_dst_flag and (fdst.level != fsrc.level)):
                # while (fdst is fsrc) or (balance_flows_flag and (fdst.level < fsrc.level)) or (eq_src_dst_flag and (fdst.level != fsrc.level)) or (burst_flag and (min(fsrc.level, fdst.level) == burst_lvl)): # ignore burst_lvl
                # fsrc = choice(hosts) # NOTE 7/22/2020: controlling flow levels now so dont need this
                # for OPT1 the if check above finds a dest that is >= so the flow level will be the source level (maybe also dest) as intended
                # for OPT2 it finds a dest of equal level so flow will be equal to src and dest as intended
                fdst = choice(hosts)
            if flows_level_method == 1:  # fixed
                f = Flow(i, fsrc, fdst, const_flow_lvl, randint(
                    min_flow_demand, max_flow_demand), durs[i - key_start])
            elif flows_level_method == 2:  # random
                f = Flow(i, fsrc, fdst, randint(min_sec_lvl, max_sec_lvl),
                         randint(min_flow_demand, max_flow_demand), durs[i - key_start])
            if flows_level_method == 3:  # == source
                f = Flow(i, fsrc, fdst, fsrc.level, randint(
                    min_flow_demand, max_flow_demand), durs[i - key_start])
            elif flows_level_method == 4:  # <= dest
                f = Flow(i, fsrc, fdst, randint(min_sec_lvl, fdst.level),
                         randint(min_flow_demand, max_flow_demand), durs[i - key_start])
            elif flows_level_method == 5:  # == min(src,dst) or <= min(src,dst)
                f = Flow(i, fsrc, fdst, min(fsrc.level, fdst.level),
                         randint(min_flow_demand, max_flow_demand), durs[i - key_start])
                # f = Flow(i, fsrc, fdst, randint(min_sec_lvl, min(fsrc.level, fdst.level)), randint(
                #     min_flow_demand, max_flow_demand), durs[i-key_start])
        flows.add(f)
    if DEBUG:
        print("Done generating flows")
    return flows


def show_graph(output_dir, g, network_type, run_type_s, optimization_problem, num_switch_values, num_flows_for_given_num_sw,
               samp_idx, m_perc, time_epoch_idx, nodesList, initial=False):
    """Display the network graph with networkx package."""
    plt.figure()
    nx_g = nx.Graph()
    nx_g.clear()
    nx_g.add_nodes_from([node.key for node in nodesList])
    nx_g2 = nx.relabel_nodes(
        nx_g, {node.key: "ID:" + node.key + "\nL:" + str(node.level) for node in nodesList})
    nx_g2.add_edges_from([("ID:" + link.k.key + "\nL:" + str(link.k.level),
                           "ID:" + link.l.key + "\nL:" + str(link.l.level)) for l, link in g["LINKS"].items()])
    nx.draw(nx_g2, with_labels=True, node_color='lightgrey',
            font_size=4, node_size=250)
    plt.savefig('%s/fig-network-topo=%s-type=%s-opt=%d-num_sw_val=%d-nflows=%d-samp=%d-M=%.2f-time_epoch=%d.pdf' % (output_dir, network_type, run_type_s,
        optimization_problem, num_switch_values, num_flows_for_given_num_sw, samp_idx, m_perc,
        time_epoch_idx if not initial else -1))
    plt.close()


def gen_mesh_topo(num_switches, fully_connected=False):
    """Generates a  meshtopology with random links and levels for nodes."""

    # Mesh topology
    hosts_dict = dict()
    switches_dict = dict()
    links_dict = dict()

    # Step 1 - Generate switches
    for sw_pos in range(1, num_switches + 1):
        sw_label = "sw" + str(sw_pos)
        if switches_level_method == 1:  # fixed
            sw_level = const_switch_lvl
        elif switches_level_method == 2:  # random
            sw_level = randint(min_sec_lvl, max_sec_lvl)
        switches_dict[sw_label] = Switch(sw_label, sw_level, switch_cap)
        host_lvl = 0
        subnet_lvl = randint(min_sec_lvl, max_sec_lvl)
        # Generate Hosts
        for host_pos in range(hosts_per_mesh_switch):
            host_label = 'h' + str((sw_pos * hosts_per_wan_switch + host_pos))
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
            hosts_dict[host_label] = Host(host_label, host_lvl)
            # Add switch-host link
            links_dict[(sw_label, host_label)] = Link(switches_dict[sw_label], hosts_dict[host_label],
                                                      host_to_edge_link_cap)
            # Add host-switch link
            links_dict[(host_label, sw_label)] = Link(hosts_dict[host_label], switches_dict[sw_label],
                                                      host_to_edge_link_cap)

    # Step 2 - Generate inter switch links  either full connected or partially connected
    if fully_connected:
        for sw_label in switches_dict.keys():
            for sw_label_1 in switches_dict.keys():
                if sw_label != sw_label_1:
                    # Link from switch 1 to switch 2
                    links_dict[(sw_label, sw_label_1)] = Link(switches_dict[sw_label], switches_dict[sw_label_1],
                                                              mesh_switch_link_cap)
                    # Link from switch 2 to switch 1
                    links_dict[(sw_label_1, sw_label)] = Link(switches_dict[sw_label_1], switches_dict[sw_label],
                                                              mesh_switch_link_cap)
    else:
        for sw_label in switches_dict.keys():
            for sw_label_1 in switches_dict.keys():
                # 0.90 probability of the link being created
                if sw_label != sw_label_1 and randint(1, 100) > 10:
                    # Link from switch 1 to switch 2
                    links_dict[(sw_label, sw_label_1)] = Link(switches_dict[sw_label], switches_dict[sw_label_1],
                                                              mesh_switch_link_cap)
                    # Link from switch 2 to switch 1
                    links_dict[(sw_label_1, sw_label)] = Link(switches_dict[sw_label_1], switches_dict[sw_label],
                                                              mesh_switch_link_cap)

    num_total_sw = num_switches
    num_hosts = num_total_sw * hosts_per_wan_switch
    switches = sorted([s for s in switches_dict.values()],
                      key=lambda s: s.key[3:])

    hosts = list(hosts_dict.values())
    nodesList = switches + hosts

    graph = {
        "SWITCHES": switches,
        "LINKS": links_dict,
        "HOSTS": hosts,
        "nodesList": nodesList
    }
    print("Done generating mesh topo")
    return graph, num_total_sw, num_hosts
