#!/usr/bin/env python3.7
#
# sim.py
#
# Description   : Simulate heuristic algorithm.
# Created by    :
# Date          : November 2019
# Last Modified : July 2020


### Imports ###
import networkx as nx
from common import extractLinkMatrix, flow_weight, delta_j, B, optimization_problem, max_sec_lvl, min_sec_lvl, flows_level_method, get_num_route_downs, get_info_type, DEBUG, num_levels, num_warmup_epochs, reboot_time, weight_of_rebooting_switch
from random import *
from time import time_ns
from copy import deepcopy
import numpy as np


### Functions ###
def computePaths(flm, tmp, flow, flows, nodesList, feasiblePaths, path):
    """Finds feasible paths using dynamic programming depth-first search."""
    row = flm[tmp.key]
    nextNode = None
    path.append(tmp)
    if tmp.key == flow.dest.key:
        feasiblePaths[flows.index(flow)].append(path)
        return
    for j in row:
        for node in nodesList:
            if node.key == j:
                nextNode = node
                break
        # Need to check if it was covered already
        if flm[tmp.key][j] == str(1) and nextNode not in path:
            # list here to create a new list object
            computePaths(flm, nextNode, flow, flows, nodesList,
                         feasiblePaths, list(path))


def findFeasiblePaths(flm, flow, flows, nodesList, feasiblePaths):
    """Starts searching for feasible paths."""
    computePaths(flm, flow.source, flow, flows, nodesList, feasiblePaths, [])
    return


def get_switch_degrees_and_neighbor_info(g, switches_pos_dict, hosts_pos_dict, failed_links):
    """Used to gather information about switches and their neighbors before running the heuristic algorithm."""
    sw_info = dict()
    hosts_info = dict()
    for link in g['LINKS'].keys():
        # need to do this since this is when we are intializing sw_info and hosts_info
        if link[0] in switches_pos_dict.keys():
            linkhead_info_type = sw_info
        elif link[0] in hosts_pos_dict.keys():
            linkhead_info_type = hosts_info
        else:
            print("bad info_type in get_switch_degrees_and_neighbor_info")
            exit(1)

        linktail_info_type_id, linktail_info_type = get_info_type(
            link[1], switches_pos_dict, hosts_pos_dict, fstr=True)
        neighbor_level = g[linktail_info_type_id][linktail_info_type[link[1]]].level

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
        if link[0] in switches_pos_dict.keys():
            linkhead_info_type = sw_info
        elif link[0] in hosts_pos_dict.keys():
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
        # conflict_val = max(flow.level - link.l.level, 0) # find switches as close in lvl as possiblew
        conflict_val = abs(flow.level - link.l.level)
        # conflict_val = 1 if (flow.level != link.l.level) else 0

        if (switches_whos_level_changed_last_run is not None) and (link.l.key in switches_whos_level_changed_last_run):
            conflict_val += weight_of_rebooting_switch
    elif optimization_problem == 2:
        # conflict_val = max(flow.level - link.l.level, 0) # find switches as close in lvl as possiblew
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


def run_modified_djik_helper(g, nodesList, flows, network_type, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run, phase=0):
    """Runs the custom shortest-path algorithm for each flow to find potential paths, and records all conflicting switches during backtracking."""
    flow_paths = dict()
    conflict_switches_for_this_flow = dict()
    unique_conflicting_switches = set()
    total_num_unique_conflicting_switches = 0
    max_conflict_at_switch = dict()
    avg_path_length = 0
    num_valid_paths = 0
    for j in flows:
        Q = []
        dist = dict()
        prev = dict()
        for node in nodesList:
            dist[node.key] = float("inf")
            prev[node.key] = None
            Q.append(node)
        dist[j.source.key] = 0
        src_info_type_id, src_info_type = get_info_type(
            j.source.key, switches_pos_dict, hosts_pos_dict, fstr=True)
        dst_info_type_id, dst_info_type = get_info_type(
            j.dest.key, switches_pos_dict, hosts_pos_dict, fstr=True)
        if DEBUG:
            print([node.key+", " for node in nodesList])
            print("==========\nid: ", j.key)
            print("source: %s (lev: %d)" %
                  (j.source.key, g[src_info_type_id][src_info_type[j.source.key]].level))
            print("dest: %s (lev: %d)" %
                  (j.dest.key, g[dst_info_type_id][dst_info_type[j.dest.key]].level))
            print("dest mem address: ", j.dest.key)
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
            u_info_type = get_info_type(u.key, sw_info, hosts_info, fstr=False)
            if u.key not in u_info_type.keys():
                # happens when running random sometimes if nodes are disconnected from graph
                continue
            for neighbor in u_info_type[u.key]['neighbors']:
                uneighbor_info_type_id, uneighbor_info_type = get_info_type(
                    neighbor, switches_pos_dict, hosts_pos_dict, fstr=True)
                neighbor_level = g[uneighbor_info_type_id][uneighbor_info_type[neighbor]].level
                # if (switches_whos_level_changed_last_run is not None) and (neighbor in switches_whos_level_changed_last_run):
                #     # NOTE (07/22/2020):  if we do this then flows will just get blocked since they cant go through still-rebooting switches, which means they are subject to being retried up to the limit (dont use this)
                #     continue  # if switch is still rebooting, dont try to go through it
                if optimization_problem == 1:
                    if ((u.key, neighbor) in g['LINKS'].keys()):
                        if phase == 1:  # find any potential path
                            if (dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]:
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
                                prev[neighbor] = u.key
                        elif phase == 2:
                            # prevent case (1) conf after resolving
                            # fine if neighbor is dest
                            if ((dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]) and (j.level <= g['LINKS'][(u.key, neighbor)].l.level):
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
                                prev[neighbor] = u.key
                        elif phase == 3:
                            # prevent case (1) and (2) conf after resolving
                            # fine if neighbor is dest
                            if ((dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]) and (j.level <= g['LINKS'][(u.key, neighbor)].l.level) and ((g['LINKS'][(u.key, neighbor)].k.level - g['LINKS'][(u.key, neighbor)].l.level) <= delta_j):
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
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
                    if ((u.key, neighbor) in g['LINKS'].keys()):
                        if phase == 1:  # find any potential path
                            if (dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]:
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
                                prev[neighbor] = u.key
                        elif phase == 3:
                            # prevent case (1) conf after resolving
                            if ((dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run)) < dist[neighbor]) and ((j.level == g['LINKS'][(u.key, neighbor)].l.level) or (neighbor == j.dest.key)):
                                # since we start at flow source, we implicitly ignore source node level, and we have the or condition to ignore the dest node once we reach that point
                                dist[neighbor] = (
                                    dist[u.key] + get_link_weight(g['LINKS'][(u.key, neighbor)], j, switches_whos_level_changed_last_run))
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
        curr_node = j.dest.key
        flow_paths[j.key].insert(0, curr_node)
        if prev[curr_node] is not None:
            last_node = curr_node
            last_node_lvl = j.dest.level
            curr_node = prev[curr_node]
            while (last_node != j.source.key):
                flow_paths[j.key].insert(0, curr_node)
                node_lvl = 0
                info_type_nd = None
                curr_node_info_type_id, curr_node_info_type = get_info_type(
                    curr_node, switches_pos_dict, hosts_pos_dict, fstr=True)
                if (curr_node_info_type_id == 'HOSTS') and (curr_node != j.source.key):
                    print("hit a host (%s) but host is not flow source" %
                          curr_node)
                    exit(1)
                else:
                    node_lvl = g[curr_node_info_type_id][curr_node_info_type[curr_node]].level

                if optimization_problem == 1:
                    if (j.level != node_lvl) and (phase == 1):
                        # case (1) conflict: if flow is higher, need to ugprade; if flow is lower, can possibly downgrade to max_level_flow
                        if (curr_node != j.source.key):  # dont try to upgrade a source
                            # only call it a conflict in this case (used in counting for coverage purposes)
                            if j.level > node_lvl:
                                conflict_switches_for_this_flow[j.key].add(
                                    curr_node)
                            if curr_node not in unique_conflicting_switches:
                                unique_conflicting_switches.add(curr_node)
                                total_num_unique_conflicting_switches += 1
                            curr_node_info_type_id, curr_node_info_type = get_info_type(
                                curr_node, sw_info, hosts_info, fstr=True)
                            curr_node_info_type[curr_node]['has_flow_violations'] = True
                    elif ((node_lvl - last_node_lvl) > delta_j) and (phase == 2):
                        # case (2) conflict: try to downgrade tail of link
                        if (last_node != j.dest.key):  # dont try to upgrade a dest
                            conflict_switches_for_this_flow[j.key].add(
                                last_node)
                            if last_node not in unique_conflicting_switches:
                                unique_conflicting_switches.add(last_node)
                                total_num_unique_conflicting_switches += 1
                            last_node_info_type = get_info_type(
                                last_node, sw_info, hosts_info, fstr=False)
                            last_node_info_type[last_node]['has_txdown_violations'] = True
                            last_node_info_type[last_node]['txdown_bad_links'].add(
                                (curr_node, last_node))
                        elif last_node == j.dest.key:
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
                        if (curr_node != j.source.key):  # dont try to upgrade a source
                            conflict_switches_for_this_flow[j.key].add(
                                curr_node)
                            if curr_node not in unique_conflicting_switches:
                                unique_conflicting_switches.add(curr_node)
                                total_num_unique_conflicting_switches += 1
                elif optimization_problem == 3:
                    if g['LINKS'][(curr_node, last_node)].k.level > g['LINKS'][(curr_node, last_node)].l.level:
                        if not ((flows_level_method == 3) and (last_node == j.dest.key)):
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
                if (flow_paths[j.key][0] == j.source.key) and (flow_paths[j.key][-1] == j.dest.key):
                    print("ROUTED")
                else:
                    # just print these so we can try and see partial paths for nonrouted flows
                    print("NOT-ROUTED")
            if (flow_paths[j.key][0] == j.source.key) and (flow_paths[j.key][-1] == j.dest.key):
                avg_path_length += len(flow_paths[j.key])
                num_valid_paths += 1
            else:
                pass
            path_str = ""
            for node in flow_paths[j.key]:
                node_info_type_id, node_info_type = get_info_type(
                    node, switches_pos_dict, hosts_pos_dict, fstr=True)
                path_str += node + \
                    ('(lev:%d)' % g[node_info_type_id]
                     [node_info_type[node]].level)
                if node != j.dest.key:
                    path_str += " -> "
            if DEBUG:
                print("flow_path for [%s]: " % j.source.key, path_str)
                print("conflict_switches_for_this_flow for [%s]: " %
                      j.source.key, conflict_switches_for_this_flow[j.key])
                print("num tx downs: %d" %
                      (get_num_tx_downs(g, j, flow_paths[j.key])))
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


def get_results(g, g_before, max_conflict_at_switch, conflict_switches_for_this_flow, unique_conflicting_switches, total_num_unique_conflicting_switches, flows, flow_paths, switches_pos_dict, switches_whos_level_changed_last):
    """Gather results from running shortest path, including the coverage, paths, blocked flows, etc."""

    # get switch results
    new_switch_levels = dict()
    num_switches_with_level_change = 0
    switches_whos_level_changed = set()
    # only None when set below at each relabeling epoch (otherwise an empty set)
    if switches_whos_level_changed_last is None:  # hits on 20, 40, etc
        level_change_delta = 0
        for sw in g['SWITCHES']:
            if DEBUG:
                print("old level of [%s]: %d -> new level of [%s]: %d" % (sw.key, g_before['SWITCHES']
                                                                          [switches_pos_dict[sw.key]].level, sw.key, g['SWITCHES'][switches_pos_dict[sw.key]].level))
            if g_before['SWITCHES'][switches_pos_dict[sw.key]].level != g['SWITCHES'][switches_pos_dict[sw.key]].level:
                new_switch_levels[sw.key] = g['SWITCHES'][switches_pos_dict[sw.key]].level
                num_switches_with_level_change += 1
                sw.wait_time_remaining = reboot_time
                switches_whos_level_changed.add(sw.key)
                level_change_delta += g_before['SWITCHES'][switches_pos_dict[sw.key]
                                                           ].level - g['SWITCHES'][switches_pos_dict[sw.key]].level
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
    print("durs left: ", [f.wait_time_remaining for f in flows])
    for j in flows:
        if (optimization_problem == 1) or (optimization_problem == 2):
            # no conflict, but maybe no path
            num_transfer_downs_for_flow[j.key] = 0
            if len(conflict_switches_for_this_flow[j.key]) == 0:
                # if path is good
                if (flow_paths[j.key][0] == j.source.key) and (flow_paths[j.key][-1] == j.dest.key):
                    if j not in flows_seen:
                        flows_seen.add(j)
                    else:
                        if DEBUG:
                            print("saw flow already: %s" % j.key)
                            print("")
                        pass
                    num_transfer_downs_for_flow[j.key] = get_num_tx_downs(
                        g, j, flow_paths[j.key])
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
                        for sw in g['SWITCHES']:
                            if sw.key in list(switches_whos_level_changed):
                                # does flow touch this switch whos level changed within the last reboot_time seconds?
                                if sw.key in flow_paths[j.key]:
                                    waiting_flows.add(j)
                                    # max of remaining time on another potential switch and current switch (other switch could actually be this one)
                                    j.wait_time_remaining = max(
                                        j.wait_time_remaining, sw.wait_time_remaining)
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
                if (flow_paths[j.key][0] == j.source.key) and (flow_paths[j.key][-1] == j.dest.key):
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


def upgrade_switch_levels_method2(g, flows, unique_conflicting_switches, flow_paths, curr_M, sw_info, switches_pos_dict, max_conflict_at_switch_before, phase=0):
    """Runs the custom relabeling method given the global unique set of conflicting switches."""
    # Step 1 - weight all conflicting switches (and weight per level)
    flows_dict = {f.key: f for f in flows}
    conf_sw_weights = {g['SWITCHES'][switches_pos_dict[sw]]
        : 0 for sw in unique_conflicting_switches}
    conf_sw_per_lvl_weights = {g['SWITCHES'][switches_pos_dict[sw]]: {lev: 0 for lev in range(
        min_sec_lvl, max_sec_lvl+1)} for sw in unique_conflicting_switches}
    sw_weights = {g['SWITCHES'][switches_pos_dict[sw]]
        : 0 for sw in switches_pos_dict.keys()}
    sw_per_lvl_weights = {g['SWITCHES'][switches_pos_dict[sw]]: {lev: 0 for lev in range(
        min_sec_lvl, max_sec_lvl+1)} for sw in switches_pos_dict.keys()}
    for sw in sw_weights.keys(): # can easily be recorded during backtracking same way (just check the dest has a prev[] and it implies that path exists)
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

                if conf_sw_per_lvl_sorted_by_weight[i][0][1] != g['SWITCHES'][switches_pos_dict[conf_sw_per_lvl_sorted_by_weight[i][0][0].key]].level:
                    switches_upgraded.add(
                        conf_sw_per_lvl_sorted_by_weight[i][0][0].key)
                    num_switches_upgraded += 1
                g['SWITCHES'][switches_pos_dict[conf_sw_per_lvl_sorted_by_weight[i]
                                                [0][0].key]].level = conf_sw_per_lvl_sorted_by_weight[i][0][1]
                upgrade_types[conf_sw_per_lvl_sorted_by_weight[i]
                              [0][0].key] = (1, conf_sw_per_lvl_sorted_by_weight[i][0][1])  # record new level
            elif (phase == 2):
                print("bad_links head: ", [
                      bad_link[0] for bad_link in sw_info[conf_sw_per_lvl_sorted_by_weight[i][0][0].key]['txdown_bad_links']])
                max_weight_incoming_sw = max([g['SWITCHES'][switches_pos_dict[bad_link[0]]] for bad_link in sw_info[conf_sw_per_lvl_sorted_by_weight[i]
                                                                                                                    [0][0].key]['txdown_bad_links']], key=lambda sw: sw_weights[sw])
                if (max_weight_incoming_sw.level - delta_j) != g['SWITCHES'][switches_pos_dict[conf_sw_per_lvl_sorted_by_weight[i][0][0].key]].level:
                    switches_upgraded.add(
                        conf_sw_per_lvl_sorted_by_weight[i][0][0].key)
                    num_switches_upgraded += 1
                # the curr_node (max_weight_incoming_sw) should already be as low as possible from case (1) fixes (we change if flow!=nodelvl), so here we just set the last_node as close to 0 as possible (ie curr_node-delta_j)
                g['SWITCHES'][switches_pos_dict[conf_sw_per_lvl_sorted_by_weight[i]
                                                [0][0].key]].level = max_weight_incoming_sw.level - delta_j
                upgrade_types[conf_sw_per_lvl_sorted_by_weight[i]
                              [0][0].key] = (2, max_weight_incoming_sw.level - delta_j)  # record new level
        elif optimization_problem == 2:
            if conf_sw_per_lvl_sorted_by_weight[i][0][1] != g['SWITCHES'][switches_pos_dict[conf_sw_per_lvl_sorted_by_weight[i][0][0].key]].level:
                switches_upgraded.add(
                    conf_sw_per_lvl_sorted_by_weight[i][0][0].key)
                num_switches_upgraded += 1
            g['SWITCHES'][switches_pos_dict[conf_sw_per_lvl_sorted_by_weight[i]
                                            [0][0].key]].level = conf_sw_per_lvl_sorted_by_weight[i][0][1]
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


def run_flow_based_heuristic(g, nodesList, flows, network_type, m_perc, time_epoch, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, relabeling_period, switches_whos_level_changed_last_run):
    """Invokes the custom shortest-path and upgrading subroutines, collects results, and returns them"""
    M = int(len(g['SWITCHES']) * m_perc)
    # switches_pos_dict = {g['SWITCHES']
    #                      [i].key: i for i in range(len(g['SWITCHES']))}
    # hosts_pos_dict = {g['HOSTS'][i].key: i for i in range(len(g['HOSTS']))}
    # sw_info, hosts_info = get_switch_degrees_and_neighbor_info(
    #     g, switches_pos_dict, hosts_pos_dict)
    flows_sorted = sorted(flows, key=lambda j: j.demand *
                          flow_weight(j.level), reverse=True)
    # if DEBUG:
    #     print("===\nFlows sorted: ", [
    #         ("key: "+str(f.key), "lev: "+str(f.level), "demand: "+str(f.demand)) for f in flows_sorted])
    g_before = deepcopy(g)

    num_sw_left_to_upgrade = M
    switches_whos_level_changed_last_run_copy = deepcopy(
        switches_whos_level_changed_last_run)

    num_switches_upgraded_phase1 = 0
    num_switches_upgraded_phase2 = 0
    if optimization_problem == 1:
        # Phase 1
        start_time = time_ns()
        max_conflict_at_switch_phase1, conflict_switches_for_this_flow_phase1, unique_conflicting_switches_phase1, total_num_unique_conflicting_switches_phase1, flow_paths_phase1, avg_path_len_phase1 = run_modified_djik_helper(
            g, nodesList, flows, network_type, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run_copy, phase=1)
        flows_routed_phase1, num_flows_routed_by_level_phase1, num_flows_routed_phase1, num_flows_phase1, blocked_flows_phase1, flows_failing_tx_down_delta_phase1, flows_failing_tx_down_num_phase1, num_transfer_downs_for_flow_phase1, num_routing_downs_for_flow_phase1, num_switches_with_level_change_phase1, switches_whos_level_changed_phase1, waiting_flows_phase1 = get_results(
            g, g_before, max_conflict_at_switch_phase1, conflict_switches_for_this_flow_phase1, unique_conflicting_switches_phase1, total_num_unique_conflicting_switches_phase1, flows, flow_paths_phase1, switches_pos_dict, switches_whos_level_changed_last_run_copy)
        # if time_epoch >= num_warmup_epochs:
        if (time_epoch >= num_warmup_epochs) and ((time_epoch - num_warmup_epochs) % relabeling_period == 0):
            num_switches_upgraded_phase1, switches_upgraded_phase1, upgrade_types_phase1, num_sw_left_to_upgrade, conf_sw_sorted_by_weight_phase1, conf_sw_per_lvl_sorted_by_weight_phase1 = upgrade_switch_levels_method2(
                g, flows, unique_conflicting_switches_phase1, flow_paths_phase1, num_sw_left_to_upgrade, sw_info, switches_pos_dict, max_conflict_at_switch_phase1, phase=1)
            # indicates to get_results to recollect the relabeled switches
            switches_whos_level_changed_last_run_copy = None

        # Phase 2
        max_conflict_at_switch_phase2, conflict_switches_for_this_flow_phase2, unique_conflicting_switches_phase2, total_num_unique_conflicting_switches_phase2, flow_paths_phase2, avg_path_len_phase2 = run_modified_djik_helper(
            g, nodesList, flows, network_type, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run_copy, phase=2)
        flows_routed_phase2, num_flows_routed_by_level_phase2, num_flows_routed_phase2, num_flows_phase2, blocked_flows_phase2, flows_failing_tx_down_delta_phase2, flows_failing_tx_down_num_phase2, num_transfer_downs_for_flow_phase2, num_routing_downs_for_flow_phase2, num_switches_with_level_change_phase2, switches_whos_level_changed_phase2, waiting_flows_phase2 = get_results(
            g, g_before, max_conflict_at_switch_phase2, conflict_switches_for_this_flow_phase2, unique_conflicting_switches_phase2, total_num_unique_conflicting_switches_phase2, flows, flow_paths_phase2, switches_pos_dict, switches_whos_level_changed_last_run_copy)
        # if time_epoch >= num_warmup_epochs:
        if (time_epoch >= num_warmup_epochs) and ((time_epoch - num_warmup_epochs) % relabeling_period == 0):
            num_switches_upgraded_phase2, switches_upgraded_phase2, upgrade_types_phase2, num_sw_left_to_upgrade, conf_sw_sorted_by_weight_phase2, conf_sw_per_lvl_sorted_by_weight_phase2 = upgrade_switch_levels_method2(
                g, flows, unique_conflicting_switches_phase2, flow_paths_phase2, num_sw_left_to_upgrade, sw_info, switches_pos_dict, max_conflict_at_switch_phase2, phase=2)
            # indicates to get_results to recollect the relabeled switches
            switches_whos_level_changed_last_run_copy = None

        # Phase 3
        max_conflict_at_switch_phase3, conflict_switches_for_this_flow_phase3, unique_conflicting_switches_phase3, total_num_unique_conflicting_switches_phase3, flow_paths_phase3, avg_path_len_phase3 = run_modified_djik_helper(
            g, nodesList, flows, network_type, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run_copy, phase=3)
        flows_routed_phase3, num_flows_routed_by_level_phase3, num_flows_routed_phase3, num_flows_phase3, blocked_flows_phase3, flows_failing_tx_down_delta_phase3, flows_failing_tx_down_num_phase3, num_transfer_downs_for_flow_phase3, num_routing_downs_for_flow_phase3, num_switches_with_level_change_phase3, switches_whos_level_changed_phase3, waiting_flows_phase3 = get_results(
            g, g_before, max_conflict_at_switch_phase3, conflict_switches_for_this_flow_phase3, unique_conflicting_switches_phase3, total_num_unique_conflicting_switches_phase3, flows, flow_paths_phase3, switches_pos_dict, switches_whos_level_changed_last_run_copy)
    elif optimization_problem == 2:
        # Phase 1
        start_time = time_ns()
        max_conflict_at_switch_phase1, conflict_switches_for_this_flow_phase1, unique_conflicting_switches_phase1, total_num_unique_conflicting_switches_phase1, flow_paths_phase1, avg_path_len_phase1 = run_modified_djik_helper(
            g, nodesList, flows, network_type, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run_copy, phase=1)
        flows_routed_phase1, num_flows_routed_by_level_phase1, num_flows_routed_phase1, num_flows_phase1, blocked_flows_phase1, flows_failing_tx_down_delta_phase1, flows_failing_tx_down_num_phase1, num_transfer_downs_for_flow_phase1, num_routing_downs_for_flow_phase1, num_switches_with_level_change_phase1, switches_whos_level_changed_phase1, waiting_flows_phase1 = get_results(
            g, g_before, max_conflict_at_switch_phase1, conflict_switches_for_this_flow_phase1, unique_conflicting_switches_phase1, total_num_unique_conflicting_switches_phase1, flows, flow_paths_phase1, switches_pos_dict, switches_whos_level_changed_last_run_copy)
        # if time_epoch >= num_warmup_epochs:
        if (time_epoch >= num_warmup_epochs) and ((time_epoch - num_warmup_epochs) % relabeling_period == 0):
            num_switches_upgraded_phase1, switches_upgraded_phase1, upgrade_types_phase1, num_sw_left_to_upgrade, conf_sw_sorted_by_weight_phase1, conf_sw_per_lvl_sorted_by_weight_phase1 = upgrade_switch_levels_method2(
                g, flows, unique_conflicting_switches_phase1, flow_paths_phase1, num_sw_left_to_upgrade, sw_info, switches_pos_dict, max_conflict_at_switch_phase1, phase=1)
            # indicates to get_results to recollect the relabeled switches
            switches_whos_level_changed_last_run_copy = None

        # Phase 3 (no case2 conflicts for opt2)
        max_conflict_at_switch_phase3, conflict_switches_for_this_flow_phase3, unique_conflicting_switches_phase3, total_num_unique_conflicting_switches_phase3, flow_paths_phase3, avg_path_len_phase3 = run_modified_djik_helper(
            g, nodesList, flows, network_type, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, switches_whos_level_changed_last_run_copy, phase=3)
        flows_routed_phase3, num_flows_routed_by_level_phase3, num_flows_routed_phase3, num_flows_phase3, blocked_flows_phase3, flows_failing_tx_down_delta_phase3, flows_failing_tx_down_num_phase3, num_transfer_downs_for_flow_phase3, num_routing_downs_for_flow_phase3, num_switches_with_level_change_phase3, switches_whos_level_changed_phase3, waiting_flows_phase3 = get_results(
            g, g_before, max_conflict_at_switch_phase3, conflict_switches_for_this_flow_phase3, unique_conflicting_switches_phase3, total_num_unique_conflicting_switches_phase3, flows, flow_paths_phase3, switches_pos_dict, switches_whos_level_changed_last_run_copy)
    else:
        print("bad opt num")
        exit(1)

    ####### Results #######
    if DEBUG:
        print("\n===== Switch results:")
    # new_switch_levels = dict()
    # num_switches_with_level_change = 0
    # switches_whos_level_changed = list()
    # level_change_delta = 0
    # for sw in g['SWITCHES']:
    #     if DEBUG:
    #         print("old level of [%s]: %d -> new level of [%s]: %d" % (sw.key, g_before['SWITCHES']
    #                                                                   [switches_pos_dict[sw.key]].level, sw.key, g['SWITCHES'][switches_pos_dict[sw.key]].level))
    #     if g_before['SWITCHES'][switches_pos_dict[sw.key]].level != g['SWITCHES'][switches_pos_dict[sw.key]].level:
    #         new_switch_levels[sw.key] = g['SWITCHES'][switches_pos_dict[sw.key]].level
    #         num_switches_with_level_change += 1
    #         switches_whos_level_changed.append(sw.key)
    #         level_change_delta += g_before['SWITCHES'][switches_pos_dict[sw.key]].level - g['SWITCHES'][switches_pos_dict[sw.key]].level
    if (DEBUG) and (time_epoch >= num_warmup_epochs):
        print("Number of switches with level change: %d" %
              num_switches_with_level_change_phase3)
        if optimization_problem == 1:
            print("Number of switches it says upgraded: %d" %
                  (num_switches_upgraded_phase1+num_switches_upgraded_phase2))
        elif optimization_problem == 2:
            print("Number of switches it says upgraded: %d" %
                  (num_switches_upgraded_phase1))
        else:
            print("bad opt num")
            exit(1)

    # if time_epoch >= num_warmup_epochs and ((time_epoch - num_warmup_epochs) % relabeling_period == 0):
    #     all_sw_upgraded = switches_upgraded_phase1
    #     if optimization_problem == 1:
    #         all_sw_upgraded = all_sw_upgraded.union(switches_upgraded_phase2)
    #     if num_switches_with_level_change != len(all_sw_upgraded):
    #         # sanity check
    #         if DEBUG:
    #             print("switches_upgraded_phase1: ", switches_upgraded_phase1)
    #             print("upgrade_types_phase1: ", upgrade_types_phase1)
    #             if optimization_problem == 1:
    #                 print("switches_upgraded_phase2: ",
    #                       switches_upgraded_phase2)
    #                 print("upgrade_types_phase2: ", upgrade_types_phase2)
    #         # kills here sometimes if same switch changed in both phases (prob changing it in phase1 then changing back in phase 2)
    #         # edit: confirmed
    #         print("this is bad")
    #         exit(1)

    if DEBUG:
        print("\n===== Host results:")
    new_host_levels = dict()
    num_hosts_with_level_change = 0
    hosts_whos_level_changed = list()
    level_change_delta = 0
    for host in g['HOSTS']:
        if DEBUG:
            print("old level of [%s]: %d -> new level of [%s]: %d" %
                  (host.key, g_before['HOSTS'][hosts_pos_dict[host.key]].level, host.key, g['HOSTS'][hosts_pos_dict[host.key]].level))
        if g_before['HOSTS'][hosts_pos_dict[host.key]].level != g['HOSTS'][hosts_pos_dict[host.key]].level:
            new_host_levels[host.key] = g['HOSTS'][hosts_pos_dict[host.key]].level
            num_hosts_with_level_change += 1
            hosts_whos_level_changed.append(host.key)
            level_change_delta += g_before['HOSTS'][hosts_pos_dict[host.key]
                                                    ].level - g['HOSTS'][hosts_pos_dict[host.key]].level
    if DEBUG:
        print("Number of hosts with level change: %d" %
              num_hosts_with_level_change)
    if num_hosts_with_level_change > 0:
        print("hosts should not change level")
        exit(1)

    # waiting_flows = list()
    # for flow in flows:
    #     if flow in flows_routed_phase3:
    #         for sw in g['SWITCHES']:
    #             if sw.key in switches_whos_level_changed:
    #                 # does flow touch this switch whos level changed?
    #                 if sw.key in flow_paths_phase3[flow.key]:
    #                     if flow.key not in waiting_flows:
    #                         # only count the flow once
    #                         waiting_flows.append(flow.key)

    # NOTE (07/20/2020): checking this in get_results now so coverages can be reflected correctly in old code in main.py
    # flows_routed_and_not_waiting_phase3 = list()
    # for flow in flows:
    #     if (flow in flows_routed_phase3) and (flow.key not in waiting_flows):
    #         for sw in g['SWITCHES']:
    #             flows_routed_and_not_waiting_phase3.append(flow)

    if (DEBUG) and (time_epoch >= num_warmup_epochs) and ((time_epoch - num_warmup_epochs) % relabeling_period == 0):
        print("\n===== Other metrics:")

        print("switches_upgraded_phase1: ", switches_upgraded_phase1)
        if optimization_problem == 1:
            print("switches_upgraded_phase2: ", switches_upgraded_phase2)

        print("max_conflict_at_switch_phase1: ", max_conflict_at_switch_phase1)
        if optimization_problem == 1:
            print("max_conflict_at_switch_phase2: ",
                  max_conflict_at_switch_phase2)

        print("conf_sw_sorted_by_weight_phase1: ",
              conf_sw_sorted_by_weight_phase1)
        print("conf_sw_per_lvl_sorted_by_weight_phase1: ",
              conf_sw_per_lvl_sorted_by_weight_phase1)
        if optimization_problem == 1:
            print("conf_sw_sorted_by_weight_phase2: ",
                  conf_sw_sorted_by_weight_phase2)
        if optimization_problem == 1:
            print("conf_sw_per_lvl_sorted_by_weight_phase2: ",
                  conf_sw_per_lvl_sorted_by_weight_phase2)

        print("unique_conflicting_switches_phase1: ",
              unique_conflicting_switches_phase1)
        print("total_num_unique_conflicting_switches_phase1: ",
              total_num_unique_conflicting_switches_phase1)
        if optimization_problem == 1:
            print("unique_conflicting_switches_phase2: ",
                  unique_conflicting_switches_phase2)
        if optimization_problem == 1:
            print("total_num_unique_conflicting_switches_phase2: ",
                  total_num_unique_conflicting_switches_phase2)

        print("Flows routed phase3: ", [f.key for f in flows_routed_phase3])
        # print("flows_routed_and_not_waiting_phase3: ",
        #       flows_routed_and_not_waiting_phase3)
        print("waiting_flows_phase3: ", [f.key for f in waiting_flows_phase3])

        if optimization_problem == 1:
            print("Num switches upgraded total (sum of phase 1 and 2) (M=%d): %d" %
                  (M, num_switches_upgraded_phase1+num_switches_upgraded_phase2))
        elif optimization_problem == 2:
            print("Num switches upgraded total (phase 1) (M=%d): %d" %
                  (M, num_switches_upgraded_phase1))
        else:
            print("bad opt num")
            exit(1)
        print("switches_upgraded_phase1: ", switches_upgraded_phase1)
        if optimization_problem == 1:
            print("switches_upgraded_phase2: ", switches_upgraded_phase2)

        print("\nAvg path length (phase1): %.2f hops" % avg_path_len_phase1)
        if optimization_problem == 1:
            print("\nAvg path length (phase2): %.2f hops" %
                  avg_path_len_phase2)
        print("\nAvg path length (phase3): %.2f hops" % avg_path_len_phase3)

    total_num_tx_downs_phase1 = np.sum([num_transfer_downs_for_flow_phase1[flow_key]
                                        for flow_key in num_transfer_downs_for_flow_phase1.keys()])
    if DEBUG:
        print("Total num tx downs (phase1): %d" % total_num_tx_downs_phase1)
    if optimization_problem == 1:
        total_num_tx_downs_phase2 = np.sum([num_transfer_downs_for_flow_phase2[flow_key]
                                            for flow_key in num_transfer_downs_for_flow_phase2.keys()])
        if DEBUG:
            print("Total num tx downs (phase2): %d" %
                  total_num_tx_downs_phase2)
    total_num_tx_downs_phase3 = np.sum([num_transfer_downs_for_flow_phase3[flow_key]
                                        for flow_key in num_transfer_downs_for_flow_phase3.keys()])
    if DEBUG:
        print("Total num tx downs (phase3): %d" % total_num_tx_downs_phase3)

    avg_num_tx_downs_phase1 = total_num_tx_downs_phase1 * 1.0 / num_flows_routed_phase1
    if DEBUG:
        print("Avg num tx downs (phase1): %.2f" % avg_num_tx_downs_phase1)

    if optimization_problem == 1:
        # contribution from flows not routed is 0 so this is fine
        avg_num_tx_downs_phase2 = total_num_tx_downs_phase2 * 1.0 / num_flows_routed_phase2
        if DEBUG:
            print("Avg num tx downs (phase2): %.2f" % avg_num_tx_downs_phase2)
    avg_num_tx_downs_phase3 = total_num_tx_downs_phase3 * 1.0 / num_flows_routed_phase3
    if DEBUG:
        print("Avg num tx downs (phase3): %.2f" % avg_num_tx_downs_phase3)

    total_num_route_downs_phase1 = np.sum(
        [num_routing_downs_for_flow_phase1[flow_key] for flow_key in num_routing_downs_for_flow_phase1.keys()])
    if DEBUG:
        print("Total num route downs (phase1): %d" %
              total_num_route_downs_phase1)
    if optimization_problem == 1:
        total_num_route_downs_phase2 = np.sum([num_routing_downs_for_flow_phase2[flow_key]
                                               for flow_key in num_routing_downs_for_flow_phase2.keys()])
        if DEBUG:
            print("Total num route downs (phase2): %d" %
                  total_num_route_downs_phase2)
    total_num_route_downs_phase3 = np.sum([num_routing_downs_for_flow_phase3[flow_key]
                                           for flow_key in num_routing_downs_for_flow_phase3.keys()])
    if DEBUG:
        print("Total num route downs (phase3): %d" %
              total_num_route_downs_phase3)

    avg_num_route_downs_phase1 = total_num_route_downs_phase1 * 1.0 / \
        num_flows_routed_phase1  # all flows routed anyway so above is fine
    if DEBUG:
        print("Avg num route downs (phase1): %.2f" %
              avg_num_route_downs_phase1)
    if optimization_problem == 1:
        avg_num_route_downs_phase2 = total_num_route_downs_phase2 * 1.0 / \
            num_flows_routed_phase2  # all flows routed anyway so above is fine
        if DEBUG:
            print("Avg num route downs (phase2): %.2f" %
                  avg_num_route_downs_phase2)
    avg_num_route_downs_phase3 = total_num_route_downs_phase3 * 1.0 / \
        num_flows_routed_phase3  # all flows routed anyway so above is fine
    if DEBUG:
        print("Avg num route downs (phase3): %.2f" %
              avg_num_route_downs_phase3)

    if DEBUG:
        print("Number of flows routed (phase1) without level changes (%d/%d): %.2f%%" %
              (num_flows_routed_phase1, num_flows_phase1, num_flows_routed_phase1*100.0/num_flows_phase1))
    if DEBUG:
        if optimization_problem == 1:
            print("Number of flows routed (phase2) with level changes (%d/%d): %.2f%%" %
                  (num_flows_routed_phase2, num_flows_phase2, num_flows_routed_phase2*100.0/num_flows_phase2))
    if DEBUG:
        print("Number of flows routed (phase3) with level changes (%d/%d): %.2f%%" %
              (num_flows_routed_phase3, num_flows_phase3, num_flows_routed_phase3*100.0/num_flows_phase3))
        # print("Number of flows routed (not waiting) (phase3) with level changes (%d/%d): %.2f%%" %
        #       (len(flows_routed_and_not_waiting_phase3), num_flows_phase3, len(flows_routed_and_not_waiting_phase3)*100.0/num_flows_phase3))

    running_time = (time_ns()-start_time)/1e9
    if DEBUG:
        print("\033[93m{}\033[00m".format(
            "[Total running time: %.3fs]" % running_time))
        print("Done heuristic")

    return flows_routed_phase1, num_flows_routed_phase1, num_flows_phase1, flows_routed_phase3, num_flows_routed_phase3, num_flows_routed_by_level_phase3, num_flows_phase3, running_time, flow_paths_phase3, conflict_switches_for_this_flow_phase3, blocked_flows_phase1, blocked_flows_phase3, flows_failing_tx_down_delta_phase1, flows_failing_tx_down_delta_phase3, flows_failing_tx_down_num_phase1, flows_failing_tx_down_num_phase3, waiting_flows_phase3, avg_num_tx_downs_phase3, total_num_tx_downs_phase3, avg_num_route_downs_phase3, total_num_route_downs_phase3, num_switches_with_level_change_phase3, switches_whos_level_changed_phase3


def run_sim(g, nodesList, flows, network_type, m_perc, time_epoch, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, relabeling_period, switches_whos_level_changed_last_run):
    """Start simulating the heuristic algorithm."""
    return run_flow_based_heuristic(g, nodesList, flows, network_type, m_perc, time_epoch, switches_pos_dict, hosts_pos_dict, sw_info, hosts_info, failed_links, relabeling_period, switches_whos_level_changed_last_run)
