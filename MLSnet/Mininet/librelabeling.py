#!/usr/bin/env python3.7
#
# sim.py
#
# Description   : Simulate heuristic algorithm.
# Created by    :
# Date          : November 2019
# Last Modified : July 2020

"""
    Heurisitc modified for easier integration/use wrt controllers

    Earlier g[SWITCHES] etc was a list of objects instead of just using the dict
    This led to additonal ds to track positions to index. 
    
    Discovery component maintains a cleaner DS (dict of dicts) so making changes to necessary methods (No change in logic)
"""

### Imports ###
import networkx as nx
# from common import extractLinkMatrix, flow_weight, delta_j, B, optimization_problem, max_sec_lvl, min_sec_lvl, flows_level_method, get_num_route_downs, get_info_type, DEBUG, num_levels, num_warmup_epochs, reboot_time, weight_of_rebooting_switch
from controller_constants import delta_j, B, optimization_problem, max_sec_lvl, min_sec_lvl, num_levels, DEBUG
from controller_constants import get_info_type, flows_level_method, HOSTS, NODES_LIST, SWITCHES, flow_weight, weight_of_rebooting_switch
from controller_constants import M_PERC
from random import *
from time import time_ns
from copy import deepcopy
import numpy as np


# Tracking pos dicts globally for performance
global_sw_info = None
global_hosts_info = None
# Dictionary of flow to valid MLS path (found by heuristic). Only populated when relabeling occurs.
valid_flow_paths_cache = {}
# Internally track switches whose level changed last run
switches_whose_level_changed_last_run=set()

### Functions ###

def find_feasible_path(network_graph, flow):
    """ Find feasible path(s) for a flow using heurisitc
    Wrapper around run_modified_djik_helper (phase 3) for two reasons:
    1. Runs per flow instead of multiple flows (Could aggreagate and invoke ?)
    2. Data structures do not map directly so we provide a simpler API
    """
    global global_sw_info,global_hosts_info
    # Any flow which was present pre relabel will be a cache hit
    if (flow.source,flow.dest) in valid_flow_paths_cache:
        # print("\t Valid flow path found in cache")
        return valid_flow_paths_cache[(flow.source,flow.dest)]

    # New flow
    flows = set()
    flows.add(flow)
    if not network_graph[NODES_LIST]:
        # Sorts by DPID
        sw_objects_list = sorted([s for s in network_graph[SWITCHES].values()],
                                 key=lambda s: s.key)

        host_object_list = list(network_graph[HOSTS].values())
        network_graph[NODES_LIST] = sw_objects_list+host_object_list

    # if not global_sw_info:
    global_sw_info, global_hosts_info = get_switch_degrees_and_neighbor_info(
        network_graph, [])

    __, __, __, __, flow_paths, __ = run_modified_djik_helper(
        network_graph, flows, global_sw_info, global_hosts_info, [], None, phase=3)
    if flow_paths:
        if (flow_paths[flow.key][0] == flow.source) and (flow_paths[flow.key][-1] == flow.dest):
            return flow_paths[flow.key]
    return None


def get_switch_degrees_and_neighbor_info(network_graph, failed_links):
    """Used to gather information about switches and their neighbors before running the heuristic algorithm."""
    sw_info = dict()
    hosts_info = dict()
    switches_dict = network_graph[SWITCHES]
    hosts_dict = network_graph[HOSTS]
    for link in network_graph['LINKS'].keys():
        # need to do this since this is when we are intializing sw_info and hosts_info
        if link[0] in switches_dict.keys():
            linkhead_info_type = sw_info
        elif link[0] in hosts_dict.keys():
            linkhead_info_type = hosts_info
        else:
            print("bad info_type in get_switch_degrees_and_neighbor_info")
            exit(1)

        linktail_info_dict_key = get_info_type(
            link[1], switches_dict, hosts_dict)
        neighbor_level = network_graph[linktail_info_dict_key][link[1]].level

        if link[0] not in linkhead_info_type.keys():  # if node not already in dict
            linkhead_info_type[link[0]] = {
                'degree': 1, 'avg_neighbor': neighbor_level, 'neighbors': {link[1]}, 'has_flow_violations': False, 'has_txdown_violations': False, 'txdown_bad_links': set()}
        else:
            linkhead_info_type[link[0]]['degree'] += 1
            linkhead_info_type[link[0]]['avg_neighbor'] += neighbor_level
            linkhead_info_type[link[0]]['neighbors'].add(link[1])

    # FIX - Add  info even for failed links the run_djk function will check if said link is in g['LINKS'].
    # If no info is available it leads to a crash. Even the failed link check expects info to be available
    for link in [l[0] for l in failed_links]:
        # need to do this since this is when we are intializing sw_info and hosts_info
        if link[0] in switches_dict.keys():
            linkhead_info_type = sw_info
        elif link[0] in hosts_dict.keys():
            linkhead_info_type = hosts_info
        else:
            print("bad info_type in get_switch_degrees_and_neighbor_info")
            exit(1)

        neighbor_level = 0

        if link[0] not in linkhead_info_type.keys():  # if node not already in dict
            linkhead_info_type[link[0]] = {
                'degree': 1, 'avg_neighbor': neighbor_level, 'neighbors': {link[1]}, 'has_flow_violations': False,
                'has_txdown_violations': False, 'txdown_bad_links': set()}
        else:
            linkhead_info_type[link[0]]['degree'] += 1
            linkhead_info_type[link[0]]['avg_neighbor'] += neighbor_level
            linkhead_info_type[link[0]]['neighbors'].add(link[1])

    for node in sw_info.keys():
        sw_info[node]['avg_neighbor'] = int(
            sw_info[node]['avg_neighbor']/sw_info[node]['degree'])
    if DEBUG:
        # print("===\nsw_info: ", sw_info)
        # print("===\nhosts_info: ", hosts_info)
        pass
    return sw_info, hosts_info


def min_dist_vertex(dist, Q):
    """Finds the next minimum distance vertex for the shortest-path algorithm."""
    min_dist = (None, float("inf"))
    for v in Q:
        if dist[v.key] < min_dist[1]:
            min_dist = (v, dist[v.key])
    if min_dist[0] is None:
        if DEBUG:
            print("its none")
        pass
    return min_dist[0]


def get_link_weight(link, flow, switches_whos_level_changed_last_run):
    """Computes the link weight given a link and the current flow being considered."""
    # if we consider negative, not sure if djikstras will still work
    if optimization_problem == 1:
        # conflict_val = max(flow.level - link.l.level, 0) # find switches as close in lvl as possible

        conflict_val = abs(flow.level - link.l.level)
        # conflict_val = 1 if (flow.level != link.l.level) else 0

        if (switches_whos_level_changed_last_run is not None) and (link.l.key in switches_whos_level_changed_last_run):
            conflict_val += weight_of_rebooting_switch
    elif optimization_problem == 2:
        # conflict_val = max(flow.level - link.l.level, 0) # find switches as close in lvl as possible
        conflict_val = abs(flow.level - link.l.level)
        # conflict_val = 1 if (flow.level != link.l.level) else 0

        if (switches_whos_level_changed_last_run is not None) and (link.l.key in switches_whos_level_changed_last_run):
            conflict_val += weight_of_rebooting_switch
    # elif optimization_problem == 3:
    #     conflict_val = 1 if (link.k.level > link.l.level) else 0
    else:
        print("bad optimization num")
        exit(1)
    return conflict_val


def run_modified_djik_helper(network_graph, flows, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run, phase=0):
    """Runs the custom shortest-path algorithm for each flow to find potential paths, and records all conflicting switches during backtracking.
        g - dictionary which contains dictionaries for switches, links, hosts and nodesList (switch+hosts)
        nodesList - switches (sorted by name apparently - WILL TRY WITHOUT FIRST) + host (list of objects)
        flows - set of flows 
        sw_info -  generated using get_switch_degrees_and_neighbor_info
        hosts_info - Same method as above
        failed_links - set
        switches_whos_level_changed_last_run - Set of keys of switches whose level changed last run
        phase - 
            1 - finds shortest paths and conflicting switches (type 1 conflicts) 
            2 - Same as 1 except this method (djik_helper) enforces a constraint to prevent type 1 conflicts and collects type 2 conflicts
            3- Finds paths enforcing constraints related to type 1 and 2 conflicts
    """

    flow_paths = dict()
    conflict_switches_for_this_flow = dict()
    unique_conflicting_switches = set()
    total_num_unique_conflicting_switches = 0
    max_conflict_at_switch = dict()
    avg_path_length = 0
    num_valid_paths = 0
    nodesList = network_graph[NODES_LIST]

    for j in flows:
        Q = []
        dist = dict()
        prev = dict()
        for node in nodesList:
            dist[node.key] = float("inf")
            prev[node.key] = None
            Q.append(node)
        dist[j.source] = 0
        src_info_dict_key = get_info_type(
            j.source, network_graph[SWITCHES], network_graph[HOSTS])
        dst_info_dict_key = get_info_type(
            j.dest, network_graph[SWITCHES], network_graph[HOSTS])
        if DEBUG:
            print([node.key+", " for node in nodesList])
            print("==========\nid: ", j.key)
            print("source: %s (lev: %d)" %
                  (j.source, network_graph[src_info_dict_key][j.source].level))
            print("dest: %s (lev: %d)" %
                  (j.dest, network_graph[dst_info_dict_key][j.dest].level))
            print("dest mem address: ", j.dest)
            print("nodesList.index: ", nodesList.index(j.dest))
            print("flow lvl: %d" % j.level)

        broke = False
        while len(Q) > 0:
            u = min_dist_vertex(dist, Q)
            if u is None:
                broke = True
                break  # sometimes u is None because min returns None, meaning the dist to nodes left in Q is inf
            Q.remove(u)
            # print(hosts_info.keys())
            u_info_type = get_info_type(
                u.key, sw_info, hosts_info, return_ds=True)
            if u.key not in u_info_type.keys():
                # happens when running random sometimes if nodes are disconnected from graph
                continue
            for neighbor in u_info_type[u.key]['neighbors']:
                uneighbor_info_dict_key = get_info_type(
                    neighbor, network_graph[SWITCHES], network_graph[HOSTS], )
                neighbor_level = network_graph[uneighbor_info_dict_key][neighbor].level
                # if (switches_whos_level_changed_last_run is not None) and (neighbor in switches_whos_level_changed_last_run):
                #     # NOTE (07/22/2020):  if we do this then flows will just get blocked since they cant go through still-rebooting switches, which means they are subject to being retried up to the limit (dont use this)
                #     continue  # if switch is still rebooting, dont try to go through it
                if optimization_problem == 1:
                    if ((u.key, neighbor) in network_graph['LINKS'].keys()):
                        if phase == 1:  # find any potential path
                            if (dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]:
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
                                prev[neighbor] = u.key
                        elif phase == 2:
                            # prevent case (1) conf after resolving
                            # fine if neighbor is dest
                            if ((dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]) and (j.level <= network_graph['LINKS'][(u.key, neighbor)].l.level):
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
                                prev[neighbor] = u.key
                        elif phase == 3:
                            # prevent case (1) and (2) conf after resolving
                            # fine if neighbor is dest
                            if ((dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]) and (j.level <= network_graph['LINKS'][(u.key, neighbor)].l.level) and ((network_graph['LINKS'][(u.key, neighbor)].k.level - network_graph['LINKS'][(u.key, neighbor)].l.level) <= delta_j):
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
                                prev[neighbor] = u.key
                        else:
                            print("bad phase num")
                            exit(1)
                    # only get the link key
                    elif (u.key, neighbor) in [l[0] for l in failed_links]:
                        print("link (%s, %s) is a simulated failed link for this time_epoch" % (
                            u.key, neighbor))
                    else:
                        print("failed_links: ", failed_links)
                        print("Link doesnt exist when it should (and not in failed links): (%s, %s)" %
                              (u.key, neighbor))
                        exit(1)
                elif optimization_problem == 2:
                    if ((u.key, neighbor) in network_graph['LINKS'].keys()):
                        if phase == 1:  # find any potential path
                            if (dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]:
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
                                prev[neighbor] = u.key
                        elif phase == 3:
                            # prevent case (1) conf after resolving
                            if ((dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]) and ((j.level == network_graph['LINKS'][(u.key, neighbor)].l.level) or (neighbor == j.dest)):
                                # since we start at flow source, we implicitly ignore source node level, and we have the or condition to ignore the dest node once we reach that point
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(network_graph['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
                                prev[neighbor] = u.key
                        else:
                            print("bad phase num")
                            exit(1)
                    # only get the link key
                    elif (u.key, neighbor) in [l[0] for l in failed_links]:
                        print("link (%s, %s) is a simulated failed link for this time_epoch" % (
                            u.key, neighbor))
                    else:
                        print("failed_links: ", failed_links)
                        print("Link doesnt exist when it should (and not in failed links): (%s, %s)" %
                              (u.key, neighbor))
                        exit(1)
                # elif optimization_problem == 3:
                #     if ((u.key, neighbor) in g['LINKS'].keys()):
                #         if (dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j)) < dist[neighbor]:
                #             dist[neighbor] = (
                #                 dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j))
                #             prev[neighbor] = u.key
                #     # only get the link key
                #     elif (u.key, neighbor) in [l[0] for l in failed_links]:
                #         print("link (%s, %s) is a simulated failed link for this time_epoch" % (
                #             u.key, neighbor))
                #     else:
                #         print("failed_links: ", failed_links)
                #         print("Link doesnt exist when it should (and not in failed links): (%s, %s)" %
                #               (u.key, neighbor))
                #         exit(1)
                else:
                    print("bad optimization num")
                    exit(1)

        if broke:
            if DEBUG:
                print("broken")

        # check each flow path to get the set of conflicting switches (and record max conflict at each switch)
        conflict_switches_for_this_flow[j.key] = set()
        flow_paths[j.key] = list()
        broke2 = False
        curr_node = j.dest
        flow_paths[j.key].insert(0, curr_node)
        if prev[curr_node] is not None:
            last_node = curr_node
            dest_node_dict_key = get_info_type(
                j.dest, network_graph[SWITCHES], network_graph[HOSTS])
            last_node_lvl = network_graph[dest_node_dict_key][j.dest].level
            curr_node = prev[curr_node]
            while (last_node != j.source):
                flow_paths[j.key].insert(0, curr_node)
                node_lvl = 0
                info_type_nd = None
                curr_node_info_dict_key = get_info_type(
                    curr_node, network_graph[SWITCHES], network_graph[HOSTS])
                if (curr_node_info_dict_key == 'HOSTS') and (curr_node != j.source):
                    print("hit a host (%s) but host is not flow source" %
                          curr_node)
                    exit(1)
                else:
                    node_lvl = network_graph[curr_node_info_dict_key][curr_node].level

                if optimization_problem == 1:
                    if (j.level != node_lvl) and (phase == 1):
                        # case (1) conflict: if flow is higher, need to ugprade; if flow is lower, can possibly downgrade to max_level_flow
                        if (curr_node != j.source):  # dont try to upgrade a source
                            # only call it a conflict in this case (used in counting for coverage purposes)
                            if j.level > node_lvl:
                                conflict_switches_for_this_flow[j.key].add(
                                    curr_node)
                            if curr_node not in unique_conflicting_switches:
                                unique_conflicting_switches.add(curr_node)
                                total_num_unique_conflicting_switches += 1
                            curr_node_info_type = get_info_type(
                                curr_node, sw_info, hosts_info, return_ds=True)
                            curr_node_info_type[curr_node]['has_flow_violations'] = True
                    elif ((node_lvl - last_node_lvl) > delta_j) and (phase == 2):
                        # case (2) conflict: try to downgrade tail of link
                        if (last_node != j.dest):  # dont try to upgrade a dest
                            conflict_switches_for_this_flow[j.key].add(
                                last_node)
                            if last_node not in unique_conflicting_switches:
                                unique_conflicting_switches.add(last_node)
                                total_num_unique_conflicting_switches += 1
                            last_node_info_type = get_info_type(
                                last_node, sw_info, hosts_info, return_ds=True)
                            last_node_info_type[last_node]['has_txdown_violations'] = True
                            last_node_info_type[last_node]['txdown_bad_links'].add(
                                (curr_node, last_node))
                        elif last_node == j.dest:
                            # if on an edge link; in this situation, we already know that head is at max_flow and we cant change the tail (host) so cant do anything, those flow will just be blocked toward that host
                            # TODO: does requeuing the head do anything?
                            pass
                    if curr_node not in max_conflict_at_switch:
                        # max ***level*** passing through this switch (unused now)
                        max_conflict_at_switch[curr_node] = j.level
                    elif j.level > max_conflict_at_switch[curr_node]:
                        max_conflict_at_switch[curr_node] = j.level

                        
                elif optimization_problem == 2:
                    if (j.level != node_lvl) and (phase == 1):
                        if (curr_node != j.source):  # dont try to upgrade a source
                            conflict_switches_for_this_flow[j.key].add(
                                curr_node)
                            if curr_node not in unique_conflicting_switches:
                                unique_conflicting_switches.add(curr_node)
                                total_num_unique_conflicting_switches += 1
                elif optimization_problem == 3:
                    if network_graph['LINKS'][(curr_node, last_node)].k.level > network_graph['LINKS'][(curr_node, last_node)].l.level:
                        if not ((flows_level_method == 3) and (last_node == j.dest)):
                            conflict_switches_for_this_flow[j.key].add(
                                last_node)
                            if last_node not in unique_conflicting_switches:
                                unique_conflicting_switches.add(last_node)
                                total_num_unique_conflicting_switches += 1
                else:
                    print("bad optimization num")
                    exit(1)

                if prev[curr_node] is None:
                    broke2 = True
                    break
                last_node = curr_node
                last_node_lvl = node_lvl
                curr_node = prev[curr_node]
        else:
            # prevent from adding the dest twice
            broke2 = True

        if not broke2:
            flow_paths[j.key].insert(0, curr_node)
        if len(flow_paths[j.key]) > 0:
            if DEBUG:
                if (flow_paths[j.key][0] == j.source) and (flow_paths[j.key][-1] == j.dest):
                    print("ROUTED")
                else:
                    # just print these so we can try and see partial paths for nonrouted flows
                    print("NOT-ROUTED")
            if (flow_paths[j.key][0] == j.source) and (flow_paths[j.key][-1] == j.dest):
                avg_path_length += len(flow_paths[j.key])
                num_valid_paths += 1
            else:
                pass
            path_str = ""
            for node in flow_paths[j.key]:
                node_info_dict_key = get_info_type(
                    node, network_graph[SWITCHES], network_graph[HOSTS])
                path_str += node + \
                    ('(lev:%d)' % network_graph[node_info_dict_key]
                     [node].level)
                if node != j.dest:
                    path_str += " -> "
            if DEBUG:
                print("flow_path for [%s]: " % j.source, path_str)
                print("conflict_switches_for_this_flow for [%s]: " %
                      j.source, conflict_switches_for_this_flow[j.key])
                print("num tx downs: %d" %
                      (get_num_tx_downs(network_graph, j, flow_paths[j.key])))
    if num_valid_paths > 0:
        avg_path_length /= num_valid_paths
    else:
        avg_path_length = 0

    return max_conflict_at_switch, conflict_switches_for_this_flow, unique_conflicting_switches, total_num_unique_conflicting_switches, flow_paths, avg_path_length


def get_num_tx_downs(g, flow, flow_path):
    """Computes the number of transfer downs along a flow path."""
    curr_node = flow_path[0]
    num_tx_downs = 0
    for i in range(1, len(flow_path)):
        if (curr_node, flow_path[i]) in g['LINKS'].keys():
            if (g['LINKS'][(curr_node, flow_path[i])].k.level - g['LINKS'][(curr_node, flow_path[i])].l.level) > 0:
                num_tx_downs += 1
        else:
            print("why is this not in links")
            exit(1)
        curr_node = flow_path[i]
    return num_tx_downs


def get_results(network_graph, network_graph_before, max_conflict_at_switch, conflict_switches_for_this_flow, unique_conflicting_switches, total_num_unique_conflicting_switches, flows, flow_paths, switches_whos_level_changed_last):
    """Gather results from running shortest path, including the coverage, paths, blocked flows, etc.
        All flows_routed etc are sets of flow objects
    """

    # get switch results
    new_switch_levels = dict()
    num_switches_with_level_change = 0
    switches_whos_level_changed = set()
    # only None when set below at each relabeling epoch (otherwise an empty set)
    if switches_whos_level_changed_last is None:  # hits on 20, 40, etc
        level_change_delta = 0
        for sw_key in network_graph[SWITCHES]:
            if DEBUG:
                print("old level of [%s]: %d -> new level of [%s]: %d" % (sw_key, network_graph_before[SWITCHES]
                                                                          [sw_key].level, sw_key, network_graph[SWITCHES][sw_key].level))
            if network_graph_before[SWITCHES][sw_key].level != network_graph[SWITCHES][sw_key].level:
                new_switch_levels[sw_key] = network_graph[SWITCHES][sw_key].level
                num_switches_with_level_change += 1
                # sw_key.wait_time_remaining = reboot_time
                switches_whos_level_changed.add(sw_key)
                level_change_delta += network_graph_before[SWITCHES][sw_key].level - \
                    network_graph[SWITCHES][sw_key].level
        pass
    else:  # hits here still on 21-39, etc
        switches_whos_level_changed = switches_whos_level_changed_last

    # get flow results
    waiting_flows = set()
    # for flow in flows:
    #     if flow in flows_routed_phase3:
    #         for sw in g['SWITCHES']:
    #             if sw.key in switches_whos_level_changed:
    #                 # does flow touch this switch whos level changed?
    #                 if sw.key in flow_paths_phase3[flow.key]:
    #                     if flow.key not in waiting_flows:
    #                         # only count the flow once
    #                         waiting_flows.append(flow.key)

    num_flows_routed = 0
    num_flows_routed_by_level = {
        level: 0 for level in range(min_sec_lvl, max_sec_lvl + 1)}
    num_flows = len(flows)
    blocked_flows = None
    flows_failing_tx_down_delta = None
    flows_failing_tx_down_num = None
    flows_routed = set()
    flows_seen = set()
    num_transfer_downs_for_flow = dict()
    num_routing_downs_for_flow = dict()
    # print("durs left: ", [f.wait_time_remaining for f in flows])
    for j in flows:
        if (optimization_problem == 1) or (optimization_problem == 2):
            # no conflict, but maybe no path
            num_transfer_downs_for_flow[j.key] = 0
            if len(conflict_switches_for_this_flow[j.key]) == 0:
                # if path is good
                if (flow_paths[j.key][0] == j.source) and (flow_paths[j.key][-1] == j.dest):
                    if j not in flows_seen:
                        flows_seen.add(j)
                    else:
                        if DEBUG:
                            print("saw flow already: %s" % j.key)
                            print("")
                        pass
                    num_transfer_downs_for_flow[j.key] = get_num_tx_downs(
                        network_graph, j, flow_paths[j.key])
                    if num_transfer_downs_for_flow[j.key] > B:  # dont add it
                        if blocked_flows is None:
                            blocked_flows = set()
                        blocked_flows.add(j)
                        if flows_failing_tx_down_num is None:
                            flows_failing_tx_down_num = set()
                        flows_failing_tx_down_num.add(j)
                    # else:  # otherwise add it
                    else:
                        is_waiting = False
                        for sw_key in network_graph[SWITCHES]:
                            if sw_key in list(switches_whos_level_changed):
                                # does flow touch this switch whos level changed within the last reboot_time seconds?
                                if sw_key in flow_paths[j.key]:
                                    waiting_flows.add(j)
                                    # max of remaining time on another potential switch and current switch (other switch could actually be this one)
                                    # j.wait_time_remaining = max(
                                    #     j.wait_time_remaining, sw_key.wait_time_remaining)
                                    is_waiting = True

                                    if blocked_flows is None:
                                        # also consider it waiting in the blocked flows set to be added to flow set in main event loop for next epoch
                                        blocked_flows = set()
                                    blocked_flows.add(j)
                        # NOTE: dont need anymore since preserving switches_whos_level_changed
                        # if (j.wait_time_remaining > 0) and (j not in waiting_flows):
                        #     # if the flow is still waiting from a previous reboot of a switch
                        #     is_waiting = True
                        #     waiting_flows.append(j)
                        if not is_waiting:  # not waiting for switch reboot
                            num_flows_routed += 1
                            num_flows_routed_by_level[j.level] += 1
                            flows_routed.add(j)
                        else:
                            pass  # already added to waiting flows above and will be couted as such
                else:  # no conflicts, but partial path that failed somewhere because of routing down
                    if blocked_flows is None:
                        blocked_flows = set()
                    blocked_flows.add(j)
                    # if no more 1/2 conf for this flow but bad path, then...
                    if flows_failing_tx_down_delta is None:
                        flows_failing_tx_down_delta = set()
                    flows_failing_tx_down_delta.add(j)
            else:  # conflicts exist for the flow
                # but the flow path is valid
                if (flow_paths[j.key][0] == j.source) and (flow_paths[j.key][-1] == j.dest):
                    if blocked_flows is None:
                        blocked_flows = set()
                    blocked_flows.add(j)
                else:  # flows have conflicts but no valid path; ie partial path with conflicts that failed routing down somewhere
                    if blocked_flows is None:
                        blocked_flows = set()
                    blocked_flows.add(j)
                    if flows_failing_tx_down_delta is None:
                        flows_failing_tx_down_delta = set()
                    flows_failing_tx_down_delta.add(j)  # ignore for opt2
        # elif (optimization_problem == 3):
        #     # just add it; no conflicts for opt3, and no limitations on # tx downs
        #     num_transfer_downs_for_flow[j.key] = get_num_tx_downs(
        #         g, j, flow_paths[j.key])
        #     num_routing_downs_for_flow[j.key] = get_num_route_downs(
        #         g, j, flow_paths[j.key])
        #     num_flows_routed += 1
        #     num_flows_routed_by_level[j.level] += 1
        #     flows_routed.append(j)
        else:
            print("bad optimization num")
            exit(1)

    return flows_routed, num_flows_routed_by_level, num_flows_routed, num_flows, blocked_flows, flows_failing_tx_down_delta, flows_failing_tx_down_num, num_transfer_downs_for_flow, num_routing_downs_for_flow, num_switches_with_level_change, switches_whos_level_changed, waiting_flows


def upgrade_switch_levels_method2(network_graph, flows, unique_conflicting_switches, flow_paths, curr_M, sw_info, max_conflict_at_switch_before, phase=0):
    """Runs the custom relabeling method given the global unique set of conflicting switches."""
    # Step 1 - weight all conflicting switches (and weight per level)
    flows_dict = {f.key: f for f in flows}
    conf_sw_weights = {network_graph['SWITCHES']
                       [sw]: 0 for sw in unique_conflicting_switches}
    conf_sw_per_lvl_weights = {network_graph['SWITCHES'][sw]: {lev: 0 for lev in range(
        min_sec_lvl, max_sec_lvl+1)} for sw in unique_conflicting_switches}
    sw_weights = {network_graph['SWITCHES'][sw] : 0 for sw in network_graph[SWITCHES].keys()}
    sw_per_lvl_weights = {network_graph[SWITCHES][sw]: {lev: 0 for lev in range(
        min_sec_lvl, max_sec_lvl+1)} for sw in network_graph[SWITCHES].keys()}
    for sw in sw_weights.keys():  # can easily be recorded during backtracking same way (just check the dest has a prev[] and it implies that path exists)
        for fkey in flow_paths.keys():
            if sw.key in flow_paths[fkey]:  # only add these to conf
                if sw.key in unique_conflicting_switches:
                    conf_sw_weights[sw] += flow_weight(flows_dict[fkey].level)
                    conf_sw_per_lvl_weights[sw][flows_dict[fkey].level] += flow_weight(
                        flows_dict[fkey].level)
                sw_weights[sw] += flow_weight(flows_dict[fkey].level)
                sw_per_lvl_weights[sw][flows_dict[fkey].level] += flow_weight(
                    flows_dict[fkey].level)

    # Step 2 -  sort by weight (most dense to least)
    conf_sw_sorted_by_weight = sorted(
        [(sw, conf_sw_weights[sw]) for sw in conf_sw_weights.keys()], key=lambda x: x[1], reverse=True)
    # conf_sw_per_lvl_sorted_by_weight = [sorted([(sw, lvl, conf_sw_per_lvl_weights[sw][lvl]) for lvl in range(
    #     min_sec_lvl, max_sec_lvl+1)], key=lambda x: x[2], reverse=True) for sw in conf_sw_weights.keys()]
    conf_sw_per_lvl_sorted_by_weight = sorted([sorted([(sw, lvl, conf_sw_per_lvl_weights[sw][lvl]) for lvl in range(
        min_sec_lvl, max_sec_lvl+1)], key=lambda x: x[2], reverse=True) for sw in conf_sw_weights.keys()], key=lambda x: x[0][2], reverse=True)  # each element of outer list is a list (of levels) and we just want to highest weighted level's tuple (index 0) and its weight (index 2)
    # conf_sw_per_lvl_sorted_by_weight = sorted([(sw, lvl, conf_sw_per_lvl_weights[sw][lvl]) for sw in conf_sw_weights.keys(
    # ) for lvl in range(min_sec_lvl, max_sec_lvl+1)], key=lambda x: x[2], reverse=True)
    sw_sorted_by_weight = sorted(
        [(sw, sw_weights[sw]) for sw in sw_weights.keys()], key=lambda x: x[1], reverse=True)  # all switches
    sw_per_lvl_sorted_by_weight = sorted([(sw, lvl, sw_per_lvl_weights[sw][lvl]) for sw in sw_weights.keys(
    ) for lvl in range(min_sec_lvl, max_sec_lvl+1)], key=lambda x: x[2], reverse=True)

    # Step 3 -  start upgrading
    num_switches_upgraded = 0
    switches_upgraded = set()
    upgrade_types = dict()
    for i in range(len(conf_sw_per_lvl_sorted_by_weight)):
        if num_switches_upgraded >= int(curr_M):
            # cant upgrade anymore
            break

        # check conflict type and upgrade
        if optimization_problem == 1:
            if (phase == 1):
                # if max_conflict_at_switch_before[conf_sw_sorted_by_weight[i][0].key] != g['SWITCHES'][switches_pos_dict[conf_sw_sorted_by_weight[i][0].key]].level:
                #     switches_upgraded.append(
                #         conf_sw_sorted_by_weight[i][0].key)
                #     num_switches_upgraded += 1
                # g['SWITCHES'][switches_pos_dict[conf_sw_sorted_by_weight[i][0].key]
                #               ].level = max_conflict_at_switch_before[conf_sw_sorted_by_weight[i][0].key]
                # upgrade_types[conf_sw_sorted_by_weight[i][0].key] = 1

                if conf_sw_per_lvl_sorted_by_weight[i][0][1] != network_graph['SWITCHES'][conf_sw_per_lvl_sorted_by_weight[i][0][0].key].level:
                    switches_upgraded.add(
                        conf_sw_per_lvl_sorted_by_weight[i][0][0].key)
                    num_switches_upgraded += 1
                network_graph['SWITCHES'][conf_sw_per_lvl_sorted_by_weight[i]
                                          [0][0].key].level = conf_sw_per_lvl_sorted_by_weight[i][0][1]
                upgrade_types[conf_sw_per_lvl_sorted_by_weight[i]
                              [0][0].key] = (1, conf_sw_per_lvl_sorted_by_weight[i][0][1])  # record new level
            elif (phase == 2):
                print("bad_links head: ", [
                      bad_link[0] for bad_link in sw_info[conf_sw_per_lvl_sorted_by_weight[i][0][0].key]['txdown_bad_links']])
                max_weight_incoming_sw = max([network_graph['SWITCHES'][bad_link[0]] for bad_link in sw_info[conf_sw_per_lvl_sorted_by_weight[i]
                                                                                                             [0][0].key]['txdown_bad_links']], key=lambda sw: sw_weights[sw])
                if (max_weight_incoming_sw.level - delta_j) != network_graph['SWITCHES'][conf_sw_per_lvl_sorted_by_weight[i][0][0].key].level:
                    switches_upgraded.add(
                        conf_sw_per_lvl_sorted_by_weight[i][0][0].key)
                    num_switches_upgraded += 1
                # the curr_node (max_weight_incoming_sw) should already be as low as possible from case (1) fixes (we change if flow!=nodelvl), so here we just set the last_node as close to 0 as possible (ie curr_node-delta_j)
                network_graph['SWITCHES'][conf_sw_per_lvl_sorted_by_weight[i]
                                          [0][0].key].level = max_weight_incoming_sw.level - delta_j
                upgrade_types[conf_sw_per_lvl_sorted_by_weight[i]
                              [0][0].key] = (2, max_weight_incoming_sw.level - delta_j)  # record new level
        elif optimization_problem == 2:
            if conf_sw_per_lvl_sorted_by_weight[i][0][1] != network_graph['SWITCHES'][conf_sw_per_lvl_sorted_by_weight[i][0][0].key].level:
                switches_upgraded.add(
                    conf_sw_per_lvl_sorted_by_weight[i][0][0].key)
                num_switches_upgraded += 1
            network_graph['SWITCHES'][conf_sw_per_lvl_sorted_by_weight[i]
                                      [0][0].key].level = conf_sw_per_lvl_sorted_by_weight[i][0][1]
            upgrade_types[conf_sw_per_lvl_sorted_by_weight[i][0][0].key] = (
                1, conf_sw_per_lvl_sorted_by_weight[i][0][1])
        # elif optimization_problem == 3:
        #     g['SWITCHES'][switches_pos_dict[conf_sw_per_lvl_sorted_by_weight[i]
        #                                     [0].key]].level = conf_sw_per_lvl_sorted_by_weight[i][1]
        #     switches_upgraded.add(
        #         conf_sw_per_lvl_sorted_by_weight[i][0].key)
        #     num_switches_upgraded += 1
        else:
            print("bad optimization num")
            exit(1)
    if DEBUG:
        print("num_switches_upgraded: ", num_switches_upgraded)
    return num_switches_upgraded, switches_upgraded, upgrade_types, curr_M-num_switches_upgraded, conf_sw_sorted_by_weight, conf_sw_per_lvl_sorted_by_weight


def run_relabel_heuristic(network_graph, flows):
    """
    Runs heuristic 
    network_graph: Dict of dicts (Switches, hosts, link and node lists)
    flows - set of flow objects
    returns active, waiting, blocked flows (dicts) and switches whose level changed (set). 
    """
    global global_sw_info, global_hosts_info
    global switches_whose_level_changed_last_run
    global valid_flow_paths_cache

    # Populate necessary DS if needed
    if not network_graph[NODES_LIST]:
    # Sorts by DPID
        sw_objects_list = sorted([s for s in network_graph[SWITCHES].values()],
                                    key=lambda s: s.key)

        host_object_list = list(network_graph[HOSTS].values())
        network_graph[NODES_LIST] = sw_objects_list+host_object_list


    global_sw_info, global_hosts_info = get_switch_degrees_and_neighbor_info(
        network_graph, [])

    # Routed flows, paths, waiting flows and paths, blocked flows (don't need paths)
    flows_routed_phase3, flow_paths_phase3, blocked_flows_phase3, waiting_flows_phase3, switches_whos_level_changed_phase3 = run_flow_based_heuristic(
        network_graph, flows, global_sw_info, global_hosts_info, [], switches_whose_level_changed_last_run)

    # Update the switch and host info (after relabeling)
    global_sw_info, global_hosts_info = get_switch_degrees_and_neighbor_info(
        network_graph, [])


    # Update switches whose level changed
    switches_whose_level_changed_last_run=deepcopy(switches_whos_level_changed_phase3)

    # Update blocked flows (Remove waiting flows)
    blocked_flows_phase3=blocked_flows_phase3-waiting_flows_phase3
    # print("Routed flows %s, blocked flows %s and waiting flows %s" %(len(flows_routed_phase3),len(blocked_flows_phase3),len(waiting_flows_phase3)))

    valid_flow_paths_cache.clear()

    active_flows_dict={}
    waiting_flows_dict={}
    blocked_flows_dict={}

    # Store feasible computed paths for routed flows and waiting flows
    for flow_object in flows_routed_phase3:
        valid_flow_paths_cache[(flow_object.source,flow_object.dest)]=flow_paths_phase3[flow_object.key]
        # print("\t Routed:",flow_paths_phase3[flow_object.key])
        active_flows_dict[(flow_object.source,flow_object.dest)]=flow_object

    for flow_object in waiting_flows_phase3:
        valid_flow_paths_cache[(flow_object.source,flow_object.dest)]=flow_paths_phase3[flow_object.key]
        # print("\t Waiting:",flow_paths_phase3[flow_object.key])
        waiting_flows_dict[(flow_object.source,flow_object.dest)]=flow_object

    for blocked_flow in blocked_flows_phase3:
        blocked_flows_dict[(blocked_flow.source,blocked_flow.dest)]=blocked_flow
        
    return active_flows_dict,waiting_flows_dict,blocked_flows_dict, switches_whos_level_changed_phase3


def run_flow_based_heuristic(network_graph, flows,  sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run):
    """Invokes the custom shortest-path and upgrading subroutines, collects results, and returns them
    RBLP involves 3 phases
    Phase 2 - Same as 1 except this method (djik_helper) enforces a constraint to prevent type 1 conflicts and collects type 2 conflicts
    Phase 3- Finds paths enforcing constraints related to type 1 and 2 conflicts
    """
    M = int(len(network_graph['SWITCHES']) * M_PERC)
   
    network_graph_before = deepcopy(network_graph)

    num_sw_left_to_upgrade = M
    switches_whos_level_changed_last_run_copy = deepcopy(
        switches_whos_level_changed_last_run)

    num_switches_upgraded_phase1 = 0
    num_switches_upgraded_phase2 = 0
    if optimization_problem == 1:
        # Phase 1 - Finds shortest paths and conflicting switches (type 1 conflicts i.e sw.level!= fw.level), the heursitic then upgrades switches
        start_time = time_ns()

        max_conflict_at_switch_phase1, conflict_switches_for_this_flow_phase1, unique_conflicting_switches_phase1, total_num_unique_conflicting_switches_phase1, flow_paths_phase1, avg_path_len_phase1 = run_modified_djik_helper(
            network_graph, flows, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run_copy, phase=1)

        flows_routed_phase1, num_flows_routed_by_level_phase1, num_flows_routed_phase1, num_flows_phase1, blocked_flows_phase1, flows_failing_tx_down_delta_phase1, flows_failing_tx_down_num_phase1, num_transfer_downs_for_flow_phase1, num_routing_downs_for_flow_phase1, num_switches_with_level_change_phase1, switches_whos_level_changed_phase1, waiting_flows_phase1 = get_results(
            network_graph, network_graph_before, max_conflict_at_switch_phase1, conflict_switches_for_this_flow_phase1, unique_conflicting_switches_phase1, total_num_unique_conflicting_switches_phase1, flows, flow_paths_phase1, switches_whos_level_changed_last_run_copy)
       
        num_switches_upgraded_phase1, switches_upgraded_phase1, upgrade_types_phase1, num_sw_left_to_upgrade, conf_sw_sorted_by_weight_phase1, conf_sw_per_lvl_sorted_by_weight_phase1 = upgrade_switch_levels_method2(
            network_graph, flows, unique_conflicting_switches_phase1, flow_paths_phase1, num_sw_left_to_upgrade, sw_info, max_conflict_at_switch_phase1, phase=1)
        # indicates to get_results to recollect the relabeled switches
        switches_whos_level_changed_last_run_copy = None

        # Phase 2
        max_conflict_at_switch_phase2, conflict_switches_for_this_flow_phase2, unique_conflicting_switches_phase2, total_num_unique_conflicting_switches_phase2, flow_paths_phase2, avg_path_len_phase2 = run_modified_djik_helper(
            network_graph, flows, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run_copy, phase=2)

        flows_routed_phase2, num_flows_routed_by_level_phase2, num_flows_routed_phase2, num_flows_phase2, blocked_flows_phase2, flows_failing_tx_down_delta_phase2, flows_failing_tx_down_num_phase2, num_transfer_downs_for_flow_phase2, num_routing_downs_for_flow_phase2, num_switches_with_level_change_phase2, switches_whos_level_changed_phase2, waiting_flows_phase2 = get_results(
            network_graph, network_graph_before, max_conflict_at_switch_phase2, conflict_switches_for_this_flow_phase2, unique_conflicting_switches_phase2, total_num_unique_conflicting_switches_phase2, flows, flow_paths_phase2, switches_whos_level_changed_last_run_copy)

        num_switches_upgraded_phase2, switches_upgraded_phase2, upgrade_types_phase2, num_sw_left_to_upgrade, conf_sw_sorted_by_weight_phase2, conf_sw_per_lvl_sorted_by_weight_phase2 = upgrade_switch_levels_method2(
            network_graph, flows, unique_conflicting_switches_phase2, flow_paths_phase2, num_sw_left_to_upgrade, sw_info, max_conflict_at_switch_phase2, phase=2)
        # indicates to get_results to recollect the relabeled switches
        switches_whos_level_changed_last_run_copy = None

        # Phase 3
        max_conflict_at_switch_phase3, conflict_switches_for_this_flow_phase3, unique_conflicting_switches_phase3, total_num_unique_conflicting_switches_phase3, flow_paths_phase3, avg_path_len_phase3 = run_modified_djik_helper(
            network_graph, flows, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run_copy, phase=3)


        flows_routed_phase3, num_flows_routed_by_level_phase3, num_flows_routed_phase3, num_flows_phase3, blocked_flows_phase3, flows_failing_tx_down_delta_phase3, flows_failing_tx_down_num_phase3, num_transfer_downs_for_flow_phase3, num_routing_downs_for_flow_phase3, num_switches_with_level_change_phase3, switches_whos_level_changed_phase3, waiting_flows_phase3 = get_results(
            network_graph, network_graph_before, max_conflict_at_switch_phase3, conflict_switches_for_this_flow_phase3, unique_conflicting_switches_phase3, total_num_unique_conflicting_switches_phase3, flows, flow_paths_phase3, switches_whos_level_changed_last_run_copy)

    
    ####### Results #######
    if DEBUG:
        print("\n===== Switch results:")
        print("Number of switches with level change: %d" %
              num_switches_with_level_change_phase3)
        if optimization_problem == 1:
            print("Number of switches it says upgraded: %d" %
                  (num_switches_upgraded_phase1+num_switches_upgraded_phase2))

        print("\n===== Other metrics:")

        print("Flows routed phase3: ", [f.key for f in flows_routed_phase3])
     
        print("waiting_flows_phase3: ", [f.key for f in waiting_flows_phase3])


    # Controller's perspective just want end result - Flows, paths and switches which need to be changed

    return flows_routed_phase3, flow_paths_phase3, blocked_flows_phase3, waiting_flows_phase3, switches_whos_level_changed_phase3


