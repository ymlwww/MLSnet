#!/usr/bin/env python3.7
#
# main.py
#
# Description   : Entry point to simulator and ILP model.
# Created by    :
# Date          : November 2019
# Last Modified : July 2020


# from random import randint
import random
import time
from copy import deepcopy
### Imports ###
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from common import gen_fat_tree_topo, gen_flows, gen_mesh_topo, gen_wan_topo_A, min_sec_lvl, \
    max_sec_lvl, optimization_problem, flow_weight, max_flow_retries, DEBUG, num_warmup_epochs, link_fails_flag, \
    link_fail_perc, \
    link_fail_period, link_fail_duration, burst_flag, burst_start_idx, link_fails_start_idx, relabeling_period, \
    reboot_time, show_sec_metric_flag, host_migration_flag, host_migration_start_idx, host_migration_period, Link, \
    host_to_edge_link_cap, host_migration_perc, measure_agility, burst_period, \
    burst_duration
from sim import run_sim, get_switch_degrees_and_neighbor_info
from solve import run_solver


### Functions ###


def release_switch_capacities(departed_flows, flow_paths_phase3, g):
    switches_pos_dict = {g['SWITCHES']
                         [i].key: i for i in range(len(g['SWITCHES']))}

    if flow_paths_phase3 is None:
        pass

    for flow in departed_flows:
        if len(flow_paths_phase3[flow.key]) > 0:
            for sw_key in flow_paths_phase3[flow.key]:
                if sw_key in switches_pos_dict.keys():
                    g['SWITCHES'][switches_pos_dict[sw_key]].capacity = g['SWITCHES'][
                                                                            switches_pos_dict[
                                                                                sw_key]].capacity + flow.demand


def update_remaining_times(flows, g, flow_paths_phase3, conflict_switches_for_this_flow_phase3,
                           switches_whos_level_changed_last_run):
    """Updates the remaining time for switches and flows."""
    print('Updating return time')
    # update switch remaining times in place
    switches_pos_dict = {g['SWITCHES']
                         [i].key: i for i in range(len(g['SWITCHES']))}
    if switches_whos_level_changed_last_run:
        for sw in deepcopy(switches_whos_level_changed_last_run):
            if g['SWITCHES'][switches_pos_dict[sw]].wait_time_remaining > 0:
                g['SWITCHES'][switches_pos_dict[sw]].wait_time_remaining -= 1
                if g['SWITCHES'][switches_pos_dict[sw]].wait_time_remaining == 0:
                    # switch done waiting, remove it so its reflected in shortest path and set -1
                    switches_whos_level_changed_last_run.remove(sw)
                    g['SWITCHES'][switches_pos_dict[sw]].wait_time_remaining = -1

    if flow_paths_phase3 is None:
        return flows, None, set()
    flows_dict = {f.key: f for f in flows}
    remaining_flows = set()
    departed_flows = set()
    for flow in flows:
        if (optimization_problem == 1) or (optimization_problem == 2):
            if (conflict_switches_for_this_flow_phase3 is not None) and (
                    len(conflict_switches_for_this_flow_phase3[flow.key]) != 0):
                # still might be empty but not routed for solver since this was just a placeholder,
                # but path empty below will indicate it was not routed
                # NOTE (7/21/2020): this if case is if the flow still has conflicts (ie blocked; see else case line 495 in sim.py) which are added in main loop after this function call
                continue
            if len(flow_paths_phase3[flow.key]) > 0:
                # was waiting last epoch (also blocked)
                if flow.wait_time_remaining > 0:
                    flow.wait_time_remaining -= 1
                    if flow.wait_time_remaining == 0:
                        # can be routed this epoch (will be realized in get_results)
                        flow.wait_time_remaining = -1
                    remaining_flows.add(flow)  # always put back in remaining
                else:  # was routed last epoch, check if it expired yet
                    # fine for solver because if flow not routed, solver leaves path empty (line 1438)
                    if (flow_paths_phase3[flow.key][0] == flow.source.key) and (
                            flow_paths_phase3[flow.key][-1] == flow.dest.key):
                        # if random.randint(0, 1) == 0:  # 50% chance to depart the flow
                        #     # if random.randint(1, 5) == 1:
                        #     # should be about same as optimal with long or short (just need bigger network) (?)
                        #     departed_flows.add(flow)
                        # else:
                        #     new_flows.add(flow)
                        flow.duration -= 1
                        if flow.duration <= 0.0:  # if expired
                            departed_flows.add(flow)
                        else:  # if still active
                            remaining_flows.add(flow)
                    else:
                        pass  # already in blocked
            else:
                # flows that weren't routed (the blocked ones) which are already added to the blocked list
                pass
        else:
            print("bad optimization num in depart")
            exit(1)

    # Let go of the switch capacity used by the departed flows
    # release_switch_capacities(departed_flows, flow_paths_phase3, g)
    return remaining_flows, departed_flows, deepcopy(switches_whos_level_changed_last_run)


def get_flow_count_by_level(flows):
    """Gets the number of flows per level."""
    flow_count_by_level = {level: 0 for level in range(
        min_sec_lvl, max_sec_lvl + 1)}
    for j in flows:
        flow_count_by_level[j.level] += 1
    return flow_count_by_level


def get_switch_count_by_level(switches):
    """Gets the number of switches per level."""
    switch_count_by_level = {
        level: 0 for level in range(min_sec_lvl, max_sec_lvl + 1)}
    for sw in switches:
        switch_count_by_level[sw.level] += 1
    return switch_count_by_level


def get_obj_val(all_flow_sets_sample_dicts_for_epoch, all_flows_routed_phase3_sample_dicts_for_epoch, num_samples):
    """Iterates through aggregate flow set across samples, for a specific epoch, and computes the average max objective value and average observed objective value.
    """
    max_obj_val_for_epoch = 0
    for samp_flow_set_idx in range(len(all_flow_sets_sample_dicts_for_epoch)):
        # for each sample flow set at this epoch
        for flow in all_flow_sets_sample_dicts_for_epoch[samp_flow_set_idx]:
            max_obj_val_for_epoch += flow.demand * flow_weight(flow.level)
    max_obj_val_for_epoch /= num_samples

    avg_obj_val_for_epoch = 0
    for samp_flow_set_idx in range(len(all_flows_routed_phase3_sample_dicts_for_epoch)):
        for flow in all_flows_routed_phase3_sample_dicts_for_epoch[samp_flow_set_idx]:
            avg_obj_val_for_epoch += flow.demand * flow_weight(flow.level)
    avg_obj_val_for_epoch /= num_samples

    return avg_obj_val_for_epoch * 100.0 / max_obj_val_for_epoch  # normalize to max obj


def update_retry_count_on_blocked_flows(blocked_flows, waiting_flows):
    """Updates the retry count for each flow, discarding any that reach the specified limit."""
    blocked_flows_still_retry = set()
    for j in blocked_flows:
        if j.retries == max_flow_retries:
            continue  # dont try again
        else:
            if j not in waiting_flows:
                j.retries += 1
            # else: # edit: dont reset to 0, only retry n times, regardless of if it retries, then gets queued, then retries again
            #     j.retries = 0
            blocked_flows_still_retry.add(j)
    return blocked_flows_still_retry


def simulate_link_fails(g):
    failed_links = set()
    temp_key_list = deepcopy(list(g['LINKS'].keys()))
    host_key_list = [h.key for h in g['HOSTS']]
    for link_key in temp_key_list:
        if random.randint(1, 100) <= (link_fail_perc * 100.0):
            # if link_key in temp_key_list:
            if (((link_key[0], link_key[1])) not in [fl[0] for fl in failed_links]) and (
                    ((link_key[1], link_key[0])) not in [fl[0] for fl in failed_links]):
                # Should not simulate edge-host link failure
                if link_key[0] not in host_key_list and link_key[1] not in host_key_list:
                    # using set but need if check to prevent trying to pop twice which gives error
                    failed_links.add(
                        ((link_key[0], link_key[1]), g['LINKS'].pop((link_key[0], link_key[1]))))
                    failed_links.add(
                        ((link_key[1], link_key[0]), g['LINKS'].pop((link_key[1], link_key[0]))))
    print("Failed links - ", failed_links)
    return failed_links


def simulate_change_host_locations(g, network_type, num_switches):
    # NOTE: if we want to have all 3 network events occuring then we would have to be aware of failed links in here too
    # pick a random host and swap all the nodes in its subnet with another subnet (do this instead of just releveling nodes which is not quite the same as migration)
    num_subnets = 0
    if network_type == 'fattree':
        num_hosts = 2 * ((num_switches / 2) ** 3)
        # divide by 2 to just count downlinks connected to hosts
        num_subnets = num_hosts / (num_switches / 2)
    elif network_type == 'wan':
        num_subnets = 25  # from ATT dataset
    elif network_type == 'mesh':
        num_subnets = num_switches

    num_subnets_migrated = 0
    hosts_migrated = set()
    while (num_subnets_migrated < int((host_migration_perc / 2.0) * num_subnets)):
        # divide by 2 since every swap affects 2 subnets so we only want to simulate a migrate half of host_migration_perc
        host1 = random.choice(g['HOSTS'])
        while host1 in hosts_migrated:
            # find new subnet
            host1 = random.choice(g['HOSTS'])
        esw1 = None
        sub1 = set()
        for link in g['LINKS'].values():
            # find incoming link (host1 -> edge switch); should only be 1 per host for all topos
            if link.k.key == host1.key:
                esw1 = link.l  # record the edge switch object
                sub1.add(link.k)  # put the host1 in the sub1 set
                hosts_migrated.add(link.k)  # or just host1
                del g['LINKS'][(link.k.key, esw1.key)]  # delete the links
                del g['LINKS'][(esw1.key, link.k.key)]
                break
        # find other hosts connected to same edge switch
        for other_host1 in g['HOSTS']:
            if ((other_host1.key, esw1.key) in g['LINKS'].keys()) and (
                    (esw1.key, other_host1.key) in g['LINKS'].keys()):
                sub1.add(other_host1)
                hosts_migrated.add(other_host1)
                del g['LINKS'][(other_host1.key, esw1.key)]  # delete the links
                del g['LINKS'][(esw1.key, other_host1.key)]

        # pick another random host from diff subnet to swap with
        host2 = random.choice(g['HOSTS'])
        while (host2 in sub1) or (host2 in hosts_migrated):
            host2 = random.choice(g['HOSTS'])
        esw2 = None
        sub2 = set()
        for link in g['LINKS'].values():
            # find incoming link (host2 -> edge switch); should only be 1 per host for all topos
            if link.k.key == host2.key:
                esw2 = link.l  # record the edge switch object
                sub2.add(link.k)  # put the host2 in the sub2 set
                hosts_migrated.add(link.k)  # or just host2
                del g['LINKS'][(link.k.key, esw2.key)]  # delete the links
                del g['LINKS'][(esw2.key, link.k.key)]
                break
        # find other hosts connected to same edge switch
        for other_host2 in g['HOSTS']:
            if ((other_host2.key, esw2.key) in g['LINKS'].keys()) and (
                    (esw2.key, other_host2.key) in g['LINKS'].keys()):
                sub2.add(other_host2)
                hosts_migrated.add(other_host2)
                del g['LINKS'][(other_host2.key, esw2.key)]  # delete the links
                del g['LINKS'][(esw2.key, other_host2.key)]

        # attach hosts from either subnet to the other edge switch
        for sub1_host in sub1:
            g['LINKS'][(sub1_host.key, esw2.key)] = Link(
                sub1_host, esw2, host_to_edge_link_cap)
            g['LINKS'][(esw2.key, sub1_host.key)] = Link(
                esw2, sub1_host, host_to_edge_link_cap)

        for sub2_host in sub2:
            g['LINKS'][(sub2_host.key, esw1.key)] = Link(
                sub2_host, esw1, host_to_edge_link_cap)
            g['LINKS'][(esw1.key, sub2_host.key)] = Link(
                esw1, sub2_host, host_to_edge_link_cap)

        num_subnets_migrated += 1

    # should change g in place so nothing to return
    return


def refresh_failed_links(g, failed_links_af_last_run):
    for link in failed_links_af_last_run:
        g['LINKS'][link[0]] = link[1]  # bidirectional links already in list


def replicate_host_migration_event(g, non_relabelling_g):
    non_relabelling_host_dict = {h.key: h for h in non_relabelling_g["HOSTS"]}
    non_relabelling_sw_dict = {sw.key: sw for sw in non_relabelling_g["SWITCHES"]}
    old_links = [link for link in non_relabelling_g["LINKS"].keys()]
    for link in old_links:
        # Find all sw to host links and delete both (sw,hs) (hs,sw)
        if link[1] in non_relabelling_host_dict.keys():
            del non_relabelling_g["LINKS"][(link[0], link[1])]
            del non_relabelling_g["LINKS"][(link[1], link[0])]

    for link in g["LINKS"].keys():
        # Find all (sw,hs) links and add both (sw,hs) (hs,sw)
        if link[1] in non_relabelling_host_dict.keys():
            node_1 = non_relabelling_sw_dict[link[0]]
            node_2 = non_relabelling_host_dict[link[1]]
            non_relabelling_g["LINKS"][(link[0], link[1])] = Link(node_1, node_2, host_to_edge_link_cap)
            non_relabelling_g["LINKS"][(link[1], link[0])] = Link(node_2, node_1, host_to_edge_link_cap)

    return


def replicate_failed_links(failed_links, non_relabelling_g):
    non_relabelling_failed_links = set()
    for link in failed_links:
        # bidirectional links already in list
        non_relabelling_failed_links.add((link[0],non_relabelling_g["LINKS"].pop(link[0])))
    return non_relabelling_failed_links


def run_exp():
    """Runs a series of experiments for either the linear program solver or the heuristic."""
    # run_types[run_type_idx] = None
    # if run_types[run_type_idx] == 1:
    #     print("[Running solver relabeling solution...]")
    #     run_type_s = "Solver"
    # elif run_types[run_type_idx] == 2:
    #     print("[Running heuristic relabeling solution...]")
    #     run_type_s = "Heuristic"
    # else:
    #     print("bad run_types[run_type_idx]")
    #     exit(1)

    OUTPUT_DIR = './tmp'
    run_types = ['Solver','Heuristic']
    network_type = 'mesh'
    num_samples = 5
    num_relabeling_epochs = 1250
    num_epochs = num_warmup_epochs + num_relabeling_epochs

    num_switch_values = [20]
    # should be divisble by # levels
    # [num_levels*i for i in range(1, 11)]
    num_flows_for_given_num_sw = [32]  # [4*i for i in range(3, 8)]
    m_percs = [0.10]  # higher M first to have better legend ordering; too crowded with multiple M's tho

    # For security metric - assuming single  M, # flows and # switches
    # TODO: make it robust wrt flows, M and # switches
    switch_downgrade_risk = {time_epoch_idx: dict()
                             for time_epoch_idx in range(num_epochs)}

    fraction_of_high_flows_per_switch_may_route_down = {time_epoch_idx: dict()
                             for time_epoch_idx in range(num_epochs)}

    fraction_of_high_flows__may_route_down_per_switch_may_route_down = {time_epoch_idx: dict()
                                         for time_epoch_idx in range(num_epochs)}

    flow_downgrade_risk = {time_epoch_idx: dict()
                           for time_epoch_idx in range(num_epochs)}

    flows_use_switch_route_down = {time_epoch_idx: dict()
                                   for time_epoch_idx in range(num_epochs)}

    fraction_of_flows_use_switch_route_down = {time_epoch_idx: dict()
                                               for time_epoch_idx in range(num_epochs)}

    fraction_of_flows_route_down_indirectly_more_than_one_level = {time_epoch_idx: dict()
                                               for time_epoch_idx in range(num_epochs)}

    if measure_agility:
        first_event_start = 0
        num_event_bursts = 0
        event_type = None
        event_duration = 0
        event_period = 0
        one_event_burst = False
        if not (host_migration_flag or link_fails_flag or burst_flag):
            print("Bad agility config - no events being simulated")
            exit(0)
        elif burst_flag:
            num_event_bursts = (num_epochs / burst_period) - (burst_start_idx / burst_period)
            first_event_start = ceil(burst_start_idx / burst_period) * burst_period
            event_duration = burst_duration
            event_period = burst_period
            event_type = "burst"
        elif link_fails_flag:
            num_event_bursts = (num_epochs - link_fails_start_idx) / link_fail_period
            first_event_start = ceil(link_fails_start_idx / link_fail_period) * link_fail_period
            event_duration = link_fail_duration
            event_period = link_fail_period
            event_type = "link failure"
        elif host_migration_flag:
            num_event_bursts = (num_epochs - host_migration_start_idx) / host_migration_period
            first_event_start = ceil(host_migration_start_idx / host_migration_period) * host_migration_period
            event_period = host_migration_period
            event_type = "host migration"
            event_duration = 6 * relabeling_period
        closest_to_burst_recovery_epoch = ceil(first_event_start / relabeling_period) * relabeling_period
        print("Closest recovery epoch -", closest_to_burst_recovery_epoch)
        num_event_bursts = ceil(num_event_bursts)
        if num_event_bursts == 0 or one_event_burst:
            num_event_bursts = 1

    else:
        first_event_start = 100
        closest_to_burst_recovery_epoch = 3 * relabeling_period

    coverages_phase1 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                         range(len(num_switch_values))] for _ in range(len(run_types))]
    coverages_phase3 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                         range(len(num_switch_values))] for _ in range(len(run_types))]
    coverages_by_level_phase3 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                  range(len(num_switch_values))] for _ in range(len(run_types))]
    flow_counts_by_level = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                             range(len(num_switch_values))] for _ in range(len(run_types))]
    switch_counts_by_level = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                               range(len(num_switch_values))] for _ in range(len(run_types))]
    all_flow_sets = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                      range(len(num_switch_values))] for _ in range(len(run_types))]
    all_flows_routed_phase3 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                range(len(num_switch_values))] for _ in range(len(run_types))]
    running_times = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                      range(len(num_switch_values))] for _ in range(len(run_types))]
    disruption_perc_flows_booted = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                     range(len(num_switch_values))] for _ in range(len(run_types))]
    disruption_perc_flows_path_changed = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                           range(len(num_switch_values))] for _ in range(len(run_types))]
    disruption_perc_flows_no_path_changed = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                              range(len(num_switch_values))] for _ in range(len(run_types))]
    disruption_perc_flows_queued = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                     range(len(num_switch_values))] for _ in range(len(run_types))]
    avg_num_tx_downs_phase3 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                range(len(num_switch_values))] for _ in range(len(run_types))]
    total_num_tx_downs_phase3 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                  range(len(num_switch_values))] for _ in range(len(run_types))]
    avg_num_route_downs_phase3 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                   range(len(num_switch_values))] for _ in range(len(run_types))]
    total_num_route_downs_phase3 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                     range(len(num_switch_values))] for _ in range(len(run_types))]

    # All three data structures are for Agility
    time_to_adapt = [[[[[None for _ in range(
        num_samples)] for _ in range(len(m_percs))] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                      range(len(num_switch_values))] for _ in range(len(run_types))]

    switches_whos_level_changed_recovery_epochs = {sample_id: {epoch_id: 0 for epoch_id in range(num_epochs)} for
                                                   sample_id in range(num_samples)}
    flows_affected_recovery_epochs = {sample_id: {epoch_id: (0, 0, 0) for epoch_id in range(num_epochs)} for sample_id
                                      in range(num_samples)}

    non_relabelling_coverages_phase3 = [[[[[[None] * num_epochs for _ in range(
        len(m_percs))] for _ in range(num_samples)] for _ in range(len(num_flows_for_given_num_sw))] for _ in
                                         range(len(num_switch_values))] for _ in range(len(run_types))]

    for num_sw_idx in range(len(num_switch_values)):
        print("[Running [num_sw_val=%d]...]" %
              (num_switch_values[num_sw_idx]))
        for num_flows_idx in range(len(num_flows_for_given_num_sw)):
            print("[Running [num_sw_val=%d, num_flows=%d]...]" %
                  (num_switch_values[num_sw_idx], num_flows_for_given_num_sw[num_flows_idx]))
            for samp_idx in range(num_samples):
                print("[Running [num_sw_val=%d, num_flows=%d, sample=%d]...]" % (
                    num_switch_values[num_sw_idx], num_flows_for_given_num_sw[num_flows_idx], samp_idx))
                # generate a topo per sample now instead of per mperc; want the solver and heuristic to use same topo to compare coverage precisely
                g_initial = None
                num_total_sw_initial = None
                num_hosts_initial = None
                if network_type == 'fattree':
                    g_initial, num_total_sw_initial, num_hosts_initial = gen_fat_tree_topo(
                        num_switch_values[
                            num_sw_idx])  # reinitialize topo for this sample for each mperc for epochs
                elif network_type == 'wan':
                    g_initial, num_total_sw_initial, num_hosts_initial = gen_wan_topo_A()
                elif network_type == 'mesh':  # NOTE: switch should be in mesh but hosts will only be connected to one swtch
                    g_initial, num_total_sw_initial, num_hosts_initial = gen_mesh_topo(
                        num_switch_values[num_sw_idx])
                else:
                    print("bad topo type")
                    exit(1)

                for run_type_idx in range(len(run_types)):
                    print("[Running [num_sw_val=%d, num_flows=%d, sample=%d, run_type=%s]...]" % (
                        num_switch_values[num_sw_idx], num_flows_for_given_num_sw[num_flows_idx], samp_idx,
                        run_types[run_type_idx]))

                    for m_perc_index in range(len(m_percs)):
                        print("[Running [num_sw_val=%d, num_flows=%d, sample=%d, run_type=%s, m_perc_idx=%d]...]" % (
                            num_switch_values[num_sw_idx], num_flows_for_given_num_sw[num_flows_idx], samp_idx,
                            run_types[run_type_idx], m_perc_index))
                        g = deepcopy(g_initial)
                        nodesList = g["nodesList"]
                        num_total_sw = num_total_sw_initial
                        num_hosts = num_hosts_initial
                        hosts = g["HOSTS"]
                        flows = set()
                        key_start_id = 0
                        flows_routed_last_run = set()
                        num_flows_routed_last_run = 0
                        departed_flows = set()
                        flows_last_run = set()
                        flow_paths_af_last_run = None
                        blocked_flows_af_last_run = set()
                        waiting_flows_af_last_run = set()
                        switches_whos_level_changed_last_run = set()
                        failed_links_af_last_run = set()
                        conflict_switches_for_this_flow_af_last_run = None

                        for time_epoch_idx in range(num_epochs):
                            print(
                                "[Running [num_sw_val=%d, num_flows=%d, sample=%d, run_type=%s, m_perc_idx=%d, time=%d]...]" % (
                                    num_switch_values[num_sw_idx], num_flows_for_given_num_sw[num_flows_idx], samp_idx,
                                    run_types[run_type_idx], m_perc_index, time_epoch_idx))
                            # if time_epoch_idx == 0:  # save the state of intitial topo
                            #     show_graph(OUTPUT_DIR, g, network_type, run_types[run_type_idx], optimization_problem,
                            #                num_switch_values[num_sw_idx],
                            #                num_flows_for_given_num_sw[num_flows_idx], samp_idx, m_percs[m_perc_index],
                            #                time_epoch_idx, nodesList, initial=True)

                            # Non relabelling - Before solving the first event  copy the complete current network state
                            if measure_agility and time_epoch_idx == first_event_start:
                                print("Copy network state")
                                non_relabelling_g = deepcopy(g)
                                non_relabelling_nodesList = non_relabelling_g["nodesList"]
                                non_relabelling_hosts = non_relabelling_g["HOSTS"]
                                non_relabelling_num_total_sw = num_total_sw_initial
                                non_relabelling_num_hosts = num_hosts_initial
                                non_relabelling_flows = set()
                                non_relabelling_key_start_id = 0
                                non_relabelling_flows_routed_last_run = set()
                                non_relabelling_num_flows_routed_last_run = 0
                                non_relabelling_departed_flows = set()
                                non_relabelling_flows_last_run = set()
                                non_relabelling_flow_paths_af_last_run = None
                                non_relabelling_blocked_flows_af_last_run = set()
                                non_relabelling_waiting_flows_af_last_run = set()
                                non_relabelling_switches_whos_level_changed_last_run = set()
                                non_relabelling_failed_links_af_last_run = set()
                                non_relabelling_conflict_switches_for_this_flow_af_last_run = None
                                # for sw in switches_pos_dict.key():
                                #     if sw in non_relabelling_g[""]

                            if (flows is not None) or (
                                    switches_whos_level_changed_last_run is not None):
                                flows, departed_flows, switches_whos_level_changed_last_run = update_remaining_times(
                                    flows, g, flow_paths_af_last_run,
                                    conflict_switches_for_this_flow_af_last_run,
                                    switches_whos_level_changed_last_run)

                            if blocked_flows_af_last_run is not None:
                                blocked_flows_af_last_run = update_retry_count_on_blocked_flows(
                                    blocked_flows_af_last_run, waiting_flows_af_last_run)
                                flows |= blocked_flows_af_last_run
                            flows |= gen_flows(
                                key_start_id, num_flows_for_given_num_sw[num_flows_idx], hosts,
                                time_epoch_idx)
                            key_start_id += num_flows_for_given_num_sw[num_flows_idx]

                            failed_links = failed_links_af_last_run
                            if link_fails_flag and (time_epoch_idx >= link_fails_start_idx) and (
                                    (time_epoch_idx % link_fail_period) == link_fail_duration):
                                # refresh the failed links before collecting info for this epoch
                                # RG - Refresh here means these links are back up
                                # refresh_called += 1
                                refresh_failed_links(
                                    g,
                                    failed_links_af_last_run)  # put those links back in; can simulate an n-epoch fail with an if check
                                failed_links = set()

                            if host_migration_flag and (time_epoch_idx >= host_migration_start_idx) and (
                                    time_epoch_idx % host_migration_period == 0):
                                if one_event_burst and time_epoch_idx == host_migration_period or not one_event_burst:
                                    # change the host locations before collecting info for this epoch
                                    simulate_change_host_locations(
                                        g, network_type, num_switch_values[num_sw_idx])

                            # get network info first (moved here from run_flow_based_heuristic so we can get info on all nodes before trying to fail links; if we fail links before then some nodes wont be in the info lists)
                            switches_pos_dict = {g['SWITCHES'][i].key: i for i in range(
                                len(g['SWITCHES']))}
                            hosts_pos_dict = {g['HOSTS'][i].key: i for i in range(
                                len(g['HOSTS']))}
                            sw_info, hosts_info = get_switch_degrees_and_neighbor_info(
                                g, switches_pos_dict, hosts_pos_dict, failed_links)

                            # simulate link fails again
                            # failed_links = set() # moved above
                            if link_fails_flag and (time_epoch_idx >= link_fails_start_idx) and (
                                    time_epoch_idx % link_fail_period == 0):
                                if (one_event_burst and time_epoch_idx == link_fail_period) or not one_event_burst:
                                    failed_links = simulate_link_fails(
                                        g)  # delete some links from g and save them to refresh next epoch

                            if (g is None) or (nodesList is None) or (
                                    flows is None) or (num_total_sw is None) or (
                                    num_hosts is None):
                                print("bad")
                                exit(1)

                            if run_types[run_type_idx] == 'Solver':
                                flows_routed_phase1, num_flows_routed_phase1, num_flows_sorted_phase1, flows_routed_phase3, num_flows_routed_phase3, num_flows_routed_phase3_by_level, num_flows_sorted_phase3, running_time, flow_paths_phase3, conflict_switches_for_this_flow_phase3, blocked_flows_phase1, blocked_flows_phase3, flows_failing_tx_down_delta_phase1, flows_failing_tx_down_delta_phase3, flows_failing_tx_down_num_phase1, flows_failing_tx_down_num_phase3, waiting_flows_phase3, \
                                avg_num_tx_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][time_epoch_idx], \
                                total_num_tx_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][time_epoch_idx], \
                                avg_num_route_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][time_epoch_idx], \
                                total_num_route_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][
                                    time_epoch_idx], num_switches_with_level_change_phase3, switches_whos_level_changed_phase3, rc = run_solver(
                                    g, nodesList, flows, network_type,
                                    m_percs[m_perc_index], time_epoch_idx, switches_pos_dict, hosts_pos_dict,
                                    failed_links,
                                    switches_whos_level_changed_last_run)
                                if rc == -1:
                                    # if solver is not optimal, just go to next mperc/sample (will have remaining epochs of some samples that will just have None entries and wont be averaged)
                                    break

                            elif run_types[run_type_idx] == 'Heuristic':
                                flows_routed_phase1, num_flows_routed_phase1, num_flows_sorted_phase1, flows_routed_phase3, num_flows_routed_phase3, num_flows_routed_phase3_by_level, num_flows_sorted_phase3, running_time, flow_paths_phase3, conflict_switches_for_this_flow_phase3, blocked_flows_phase1, blocked_flows_phase3, flows_failing_tx_down_delta_phase1, flows_failing_tx_down_delta_phase3, flows_failing_tx_down_num_phase1, flows_failing_tx_down_num_phase3, waiting_flows_phase3, \
                                avg_num_tx_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][time_epoch_idx], \
                                total_num_tx_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][time_epoch_idx], \
                                avg_num_route_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][time_epoch_idx], \
                                total_num_route_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][
                                    time_epoch_idx], num_switches_with_level_change_phase3, switches_whos_level_changed_phase3 = run_sim(
                                    g, nodesList, flows, network_type,
                                    m_percs[m_perc_index], time_epoch_idx, switches_pos_dict, hosts_pos_dict, sw_info,
                                    hosts_info, failed_links, relabeling_period,
                                    switches_whos_level_changed_last_run)

                            coverages_phase1[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                time_epoch_idx] = num_flows_routed_phase1 / \
                                                  num_flows_sorted_phase1 * 100
                            coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                time_epoch_idx] = num_flows_routed_phase3 / \
                                                  num_flows_sorted_phase3 * 100
                            flow_counts_by_level[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                time_epoch_idx] = get_flow_count_by_level(
                                flows)
                            switch_counts_by_level[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                time_epoch_idx] = get_switch_count_by_level(
                                g['SWITCHES'])
                            all_flow_sets[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                time_epoch_idx] = deepcopy(
                                flows)
                            all_flows_routed_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                time_epoch_idx] = deepcopy(
                                flows_routed_phase3)
                            coverages_by_level_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                time_epoch_idx] = {
                                lev: None if (flow_counts_by_level[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                                  m_perc_index][time_epoch_idx][
                                                  lev] == 0) else (num_flows_routed_phase3_by_level[lev] * 100.0 /
                                                                   flow_counts_by_level[run_type_idx][num_sw_idx][
                                                                       num_flows_idx][samp_idx][
                                                                       m_perc_index][time_epoch_idx][lev]) for lev in
                                num_flows_routed_phase3_by_level}
                            running_times[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                time_epoch_idx] = running_time

                            if DEBUG:
                                print(
                                    "\n==================================================================")
                                print("============== Main Results (sample: %d, epoch: %d) ==============" % (
                                    samp_idx, time_epoch_idx))
                                print("Flow coverage (phase1) [sample=%d, m_perc_idx=%d] (%d/%d) = %.2f%%" %
                                      (samp_idx, m_perc_index, num_flows_routed_phase1, num_flows_sorted_phase1,
                                       coverages_phase1[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                           m_perc_index][time_epoch_idx]))
                                print("num_switches_with_level_change_phase3: ",
                                      num_switches_with_level_change_phase3)
                                print("Flows[m_perc_index=%d] routed phase1: " % (m_perc_index), [
                                    f.key for f in flows_routed_phase1])
                                if blocked_flows_phase1 is not None and len(blocked_flows_phase1) > 0:
                                    print(
                                        "Flows[m_perc_index=%d] blocked from conflicts/waiting percentage (phase1) (%d/%d) = %.2f%%" %
                                        (m_perc_index, len(blocked_flows_phase1), num_flows_sorted_phase1,
                                         len(blocked_flows_phase1) * 100.0 / num_flows_sorted_phase1))
                                    print(
                                        "Flows[m_perc_index=%d] blocked from conflicts/waiting (phase1): " % (
                                            m_perc_index),
                                        [f.key for f in blocked_flows_phase1])
                                if flows_failing_tx_down_delta_phase1 is not None and len(
                                        flows_failing_tx_down_delta_phase1) > 0:
                                    print(
                                        "Flows[m_perc_index=%d] failing tx down delta percentage (phase1) (%d/%d) = %.2f%%" %
                                        (m_perc_index, len(flows_failing_tx_down_delta_phase1), num_flows_sorted_phase1,
                                         len(flows_failing_tx_down_delta_phase1) * 100.0 / num_flows_sorted_phase1))
                                    print("Flows[m_perc_index=%d] failing tx down delta (phase1): " % (m_perc_index), [
                                        f.key for f in flows_failing_tx_down_delta_phase1])
                                if flows_failing_tx_down_num_phase1 is not None and len(
                                        flows_failing_tx_down_num_phase1) > 0:
                                    print(
                                        "Flows[m_perc_index=%d] failing tx down num percentage (phase1) (%d/%d) = %.2f%%" %
                                        (m_perc_index, len(flows_failing_tx_down_num_phase1), num_flows_sorted_phase1,
                                         len(flows_failing_tx_down_num_phase1) * 100.0 / num_flows_sorted_phase1))
                                    print("Flows[m_perc_index=%d] failing tx down num (phase1): " % (m_perc_index), [
                                        f.key for f in flows_failing_tx_down_num_phase1])

                                print("Flow coverage (phase3) [sample=%d, m_perc_idx=%d] (%d/%d) = %.2f%%" %
                                      (samp_idx, m_perc_index, num_flows_routed_phase3, num_flows_sorted_phase3,
                                       coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                           m_perc_index][time_epoch_idx]))
                                print("Flows[m_perc_index=%d] routed phase3: " % (m_perc_index), [
                                    f.key for f in flows_routed_phase3])
                                if blocked_flows_phase3 is not None and len(blocked_flows_phase3) > 0:
                                    print(
                                        "Flows[m_perc_index=%d] blocked from conflicts/waiting percentage (phase3) (%d/%d) = %.2f%%" %
                                        (m_perc_index, len(blocked_flows_phase3), num_flows_sorted_phase3,
                                         len(blocked_flows_phase3) * 100.0 / num_flows_sorted_phase3))
                                    print(
                                        "Flows[m_perc_index=%d] blocked from conflicts/waiting (phase3): " % (
                                            m_perc_index),
                                        [f.key for f in blocked_flows_phase3])
                                if flows_failing_tx_down_delta_phase3 is not None and len(
                                        flows_failing_tx_down_delta_phase3) > 0:
                                    print(
                                        "Flows[m_perc_index=%d] failing tx down delta percentage (phase3) (%d/%d) = %.2f%%" %
                                        (m_perc_index, len(flows_failing_tx_down_delta_phase3), num_flows_sorted_phase3,
                                         len(flows_failing_tx_down_delta_phase3) * 100.0 / num_flows_sorted_phase3))
                                    print("Flows[m_perc_index=%d] failing tx down delta (phase3): " % (m_perc_index), [
                                        f.key for f in flows_failing_tx_down_delta_phase3])
                                if flows_failing_tx_down_num_phase3 is not None and len(
                                        flows_failing_tx_down_num_phase3) > 0:
                                    print(
                                        "Flows[m_perc_index=%d] failing tx down num percentage (phase3) (%d/%d) = %.2f%%" %
                                        (m_perc_index, len(flows_failing_tx_down_num_phase3), num_flows_sorted_phase3,
                                         len(flows_failing_tx_down_num_phase3) * 100.0 / num_flows_sorted_phase3))
                                    print("Flows[m_perc_index=%d] failing tx down num (phase3): " % (m_perc_index), [
                                        f.key for f in flows_failing_tx_down_num_phase3])

                                print("Num flows[m_perc_index=%d] waiting now (%d/%d) = %.2f%%" % (m_perc_index, len(
                                    waiting_flows_phase3), len(flows), len(
                                    waiting_flows_phase3) * 100.0 / len(flows)))
                                print("waiting_flows_phase3: ", [
                                    f.key for f in waiting_flows_phase3])

                                print("\nTotal running time: %.3fs" %
                                      running_times[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                          time_epoch_idx])
                                # print("Number of switches: %d" %
                                #       num_total_sw)
                                print("Number of switches (num_switch_values[num_sw_idx]): %d" %
                                      num_switch_values[num_sw_idx])

                                print("Number of hosts: %d" %
                                      num_hosts)

                                print("\n========== Disruption metrics (sample: %d, epoch: %d) ==========" % (
                                    samp_idx, time_epoch_idx))
                                if num_flows_routed_last_run > 0:
                                    print("Num flows[m_perc_index=%d] routed last run (%d/%d): %.2f%%" % (
                                        m_perc_index, num_flows_routed_last_run,
                                        len(
                                            flows_last_run),
                                        num_flows_routed_last_run * 100.0 / len(
                                            flows_last_run)))
                                    print("Flows[m_perc_index=%d] routed last run: " % (m_perc_index),
                                          [fl.key for fl in flows_routed_last_run])

                                    print("\nNum flows[m_perc_index=%d] departed from last run (%d/%d): %.2f%%" % (
                                        m_perc_index, len(
                                            departed_flows),
                                        num_flows_routed_last_run,
                                        len(
                                            departed_flows) * 100.0 / num_flows_routed_last_run))
                                    print("Flows[m_perc_index=%d] departed from last run: " % (m_perc_index), [
                                        flow.key for flow in departed_flows])

                            num_booted_flows = 0
                            flows_booted = set()
                            num_paths_changed = 0
                            flows_with_path_changed = set()
                            num_flows_still_routed = 0
                            for flow in flows_routed_last_run:
                                if flow not in flows_routed_phase3:
                                    # might not be in those routed because it just departed instead
                                    if (flow not in departed_flows) and (flow not in waiting_flows_phase3):
                                        num_booted_flows += 1
                                        flows_booted.add(flow.key)
                                else:  # the flow is routed now still
                                    num_flows_still_routed += 1
                                    for ix in range(len(flow_paths_af_last_run[flow.key])):
                                        # all nodes should be same if path doesnt change
                                        if flow_paths_af_last_run[flow.key][ix] != \
                                                flow_paths_phase3[flow.key][ix]:
                                            num_paths_changed += 1  # the flow was preempted with a path change
                                            flows_with_path_changed.add(
                                                flow.key)
                                            break

                            if (departed_flows is not None) and (
                                    num_flows_routed_last_run - len(departed_flows)) > 0:
                                if DEBUG:
                                    print(
                                        "\nNum routed flows[m_perc_index=%d] disrupted from last run (booted) (%d/%d): %.2f%%" % (
                                            m_perc_index, num_booted_flows, num_flows_routed_last_run -
                                            len(departed_flows),
                                            num_booted_flows * 100.0 / (
                                                    num_flows_routed_last_run - len(
                                                departed_flows))))  # the difference is the number of flows that did not depart; ratio is number of flows that didnt depart that were disrupted
                                    print(
                                        "Flows[m_perc_index=%d] (routed flows) disrupted from last run (booted): " % (
                                            m_perc_index), flows_booted)
                                disruption_perc_flows_booted[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][
                                    time_epoch_idx] = num_booted_flows * 100.0 / (
                                        num_flows_routed_last_run - len(departed_flows))

                                if DEBUG:
                                    # routed still but path changed (only matters for those that were and are routed)
                                    print(
                                        "\nNum routed flows[m_perc_index=%d] disrupted from last run (path changed) (%d/%d): %.2f%%" % (
                                            m_perc_index, num_paths_changed,
                                            num_flows_routed_last_run - len(
                                                departed_flows),
                                            num_paths_changed * 100.0 / (
                                                    num_flows_routed_last_run - len(
                                                departed_flows))))  # the difference is the number of flows that did not depart; ratio is number of flows that didnt depart that were disrupted
                                    print(
                                        "Flows[m_perc_index=%d] (routed flows) disrupted from last run (path changed): " % (
                                            m_perc_index),
                                        flows_with_path_changed)
                                disruption_perc_flows_path_changed[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                    m_perc_index][
                                    time_epoch_idx] = num_paths_changed * 100.0 / (
                                        num_flows_routed_last_run - len(departed_flows))

                                if DEBUG:
                                    print(
                                        "\nNum flows[m_perc_index=%d] not disrupted (no path changed) (%d/%d): %.2f%%" % (
                                            m_perc_index, (num_flows_routed_last_run -
                                                           len(departed_flows)) - num_paths_changed,
                                            (num_flows_routed_last_run - len(
                                                departed_flows)),
                                            ((num_flows_routed_last_run - len(
                                                departed_flows)) - num_paths_changed) * 100.0 / (
                                                    num_flows_routed_last_run - len(
                                                departed_flows))))
                                disruption_perc_flows_no_path_changed[run_type_idx][num_sw_idx][num_flows_idx][
                                    samp_idx][m_perc_index][
                                    time_epoch_idx] = (
                                                              (
                                                                      num_flows_routed_last_run - len(
                                                                  departed_flows)) - num_paths_changed) * 100.0 / (
                                                              num_flows_routed_last_run - len(
                                                          departed_flows))

                                if DEBUG:
                                    print("\nNum flows[m_perc_index=%d] routed now (%d/%d): %.2f%%" % (
                                        m_perc_index, num_flows_routed_phase3, len(
                                            flows),
                                        num_flows_routed_phase3 * 100.0 / len(flows)))
                                    print("Flows[m_perc_index=%d] routed now: ", [
                                        fl.key for fl in flows_routed_phase3])

                            disruption_perc_flows_queued[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                m_perc_index][time_epoch_idx] = len(
                                waiting_flows_phase3) * 100.0 / len(flows)
                            print("\nNum flows[m_perc_index=%d] waiting now (%d/%d) = %.2f%%" % (
                                m_perc_index, len(waiting_flows_phase3), len(
                                    flows), len(waiting_flows_phase3) * 100.0 / len(flows)))

                            print("")

                            # NOTE (07/20/2020): record some info for these flows to simulate the waiting
                            # flows_dict = {f.key: f for f in flows}
                            # NOTE: edit: dont in get_results
                            # for flow in waiting_flows_phase3:
                            #     # set newly waiting flows (already waiting flows will be updated next iteration)
                            #     if flow.wait_time_remaining == -1:
                            #         flow.wait_time_remaining = reboot_time

                            """ Security Metric - Downgrade risk 
                                TODO - Add weight function
                            """
                            if (optimization_problem == 1 or optimization_problem == 3) and (show_sec_metric_flag):

                                """
                                For switch risks, can you tell me the average fraction of 
                                high flows processed by a switch that can route down.  If route down more than one level ever?

                                For flow risks, can you tell me fraction of flows that use a switch may route down?  

                                For indirect, can you tell me what fraction of flows may be routed down more than one level (indirectly)?
                                """
                                print("COMPUTING METRICS")
                                switch_flows_dict = {
                                    sw.key: list() for sw in g['SWITCHES']}
                                min_switch_level = {
                                    sw.key: max_sec_lvl for sw in g['SWITCHES']}
                                min_switch_down_level = dict()

                                flow_downgrade_risk[time_epoch_idx][samp_idx] = dict()
                                switch_downgrade_risk[time_epoch_idx][samp_idx] = dict()
                                fraction_of_high_flows_per_switch_may_route_down[time_epoch_idx][samp_idx]=dict()
                                fraction_of_high_flows__may_route_down_per_switch_may_route_down[time_epoch_idx][samp_idx]=dict()
                                flows_use_switch_route_down[time_epoch_idx][samp_idx]=dict()
                                fraction_of_flows_use_switch_route_down[time_epoch_idx][samp_idx] = 0

                                indirect_flow_risk={flow.key:0 for flow in flows_routed_phase3}
                                fraction_of_flows_route_down_indirectly_more_than_one_level[time_epoch_idx][samp_idx]=0
                                total_high_level_flows = 0
                                total_high_level_flows_2= 0

                                for flow in flows_routed_phase3:
                                    if flow.level!=min_sec_lvl:
                                        total_high_level_flows = total_high_level_flows+1
                                    if flow.level==max_sec_lvl or flow.level==max_sec_lvl-1:
                                        total_high_level_flows_2=total_high_level_flows_2+1

                                print("Total high level flows:", total_high_level_flows," ", total_high_level_flows_2)
                                # Computing flow downgrade risk
                                for flow in flows_routed_phase3:
                                    for sw in g['SWITCHES']:
                                        min_switch_down_level[(
                                            flow.key, sw.key)] = flow.level

                                routed_flows_sorted = sorted(
                                    flows_routed_phase3, key=lambda flow: flow.level)

                                for flow in routed_flows_sorted:
                                    next_node = None
                                    for node in flow_paths_phase3[flow.key][::-1]:
                                        if node in switch_flows_dict:
                                            switch_flows_dict[node].append(
                                                flow)
                                            min_switch_level[node] = min(
                                                min_switch_level[node], flow.level)
                                            if next_node in switch_flows_dict.keys():
                                                min_switch_down_level[(flow.key, node)] = min(
                                                    min_switch_level[next_node],
                                                    min_switch_down_level[(flow.key, node)])
                                        next_node = node

                                for flow in flows_routed_phase3:
                                    flow_downgrade_risk[time_epoch_idx][samp_idx][flow] = 0
                                    for node in flow_paths_phase3[flow.key]:
                                        if node in switch_flows_dict.keys():
                                            for lower_level_flow in switch_flows_dict[node]:
                                                if lower_level_flow.level < flow.level:
                                                    direct_flow_risk = (
                                                            flow.level - lower_level_flow.level)
                                                    indirect_flow_risk[flow.key] = (
                                                            lower_level_flow.level - min_switch_down_level[
                                                        (lower_level_flow.key, node)])
                                                    flow_downgrade_risk[time_epoch_idx][samp_idx][flow.key] = \
                                                        flow_downgrade_risk[
                                                            time_epoch_idx][samp_idx][
                                                            flow] + direct_flow_risk + indirect_flow_risk[flow.key]

                                print("Flow downgrade risk: ", flow_downgrade_risk[time_epoch_idx])

                                # Computing switch downgrade risk
                                for sw in g['SWITCHES']:
                                    switch_downgrade_risk[time_epoch_idx][samp_idx][sw.key] = 0
                                    fraction_of_high_flows_per_switch_may_route_down[time_epoch_idx][samp_idx][sw.key] = 0
                                    fraction_of_high_flows__may_route_down_per_switch_may_route_down[time_epoch_idx][samp_idx][sw.key] = 0
                                    switch_level_flow_count = {
                                        level: 0 for level in range(min_sec_lvl, max_sec_lvl + 1)}
                                    for flow in switch_flows_dict[sw.key]:
                                        switch_level_flow_count[flow.level] = switch_level_flow_count[flow.level] + 1
                                    for level in range(min_sec_lvl + 1, max_sec_lvl + 1):
                                        for lower_level in range(min_sec_lvl, level):
                                            switch_downgrade_risk[time_epoch_idx][samp_idx][sw.key] = \
                                                switch_downgrade_risk[time_epoch_idx][samp_idx][sw.key] + (
                                                        switch_level_flow_count[level] * (level - lower_level))
                                    if switch_downgrade_risk [time_epoch_idx][samp_idx][sw.key]>0:
                                        fraction_of_high_flows_per_switch_may_route_down[time_epoch_idx][samp_idx][sw.key]= (switch_level_flow_count[max_sec_lvl]+switch_level_flow_count[max_sec_lvl-1] + switch_level_flow_count[max_sec_lvl-2])/total_high_level_flows
                                    print("Switch (%s)'s downgrade risk -  %s" %
                                          (sw.key, switch_downgrade_risk[time_epoch_idx][samp_idx][sw.key]))

                                # Compute the flow related fractions
                                for flow in routed_flows_sorted:
                                    flows_use_switch_route_down[time_epoch_idx][samp_idx][flow.key]=0
                                    for node in flow_paths_phase3[flow.key]:
                                        # Update count flows that uses switches that downgrade to a lower level than the flow
                                        if node in switch_flows_dict.keys() and switch_downgrade_risk[time_epoch_idx][samp_idx][node]>0 and min_switch_down_level[(flow.key,node)]<flow.level:
                                            flows_use_switch_route_down[time_epoch_idx][samp_idx][flow.key]+=1
                                            # Counting each high level flow that passes through a switch that may downgrade it
                                            fraction_of_high_flows__may_route_down_per_switch_may_route_down[time_epoch_idx][samp_idx][node]=fraction_of_high_flows__may_route_down_per_switch_may_route_down[time_epoch_idx][samp_idx][node]+1
                                    if flows_use_switch_route_down[time_epoch_idx][samp_idx][flow.key]>0:
                                        fraction_of_flows_use_switch_route_down[time_epoch_idx][samp_idx]=fraction_of_flows_use_switch_route_down[time_epoch_idx][samp_idx]+1
                                    # Note by my definition of indirect risk it's already two levels down
                                    if indirect_flow_risk[flow.key]>0:
                                        print("Indirect flow downgrade risk prone flow of  level - ", flow.level)
                                        fraction_of_flows_route_down_indirectly_more_than_one_level[time_epoch_idx][samp_idx]  = fraction_of_flows_route_down_indirectly_more_than_one_level[time_epoch_idx][samp_idx] + 1

                                if total_high_level_flows>0:
                                    fraction_of_flows_use_switch_route_down[time_epoch_idx][samp_idx]=fraction_of_flows_use_switch_route_down[time_epoch_idx][samp_idx]/len(flows_routed_phase3)
                                    for sw in g['SWITCHES']:
                                        fraction_of_high_flows__may_route_down_per_switch_may_route_down[time_epoch_idx][
                                            samp_idx][sw.key] = fraction_of_high_flows__may_route_down_per_switch_may_route_down[time_epoch_idx][samp_idx][sw.key]/total_high_level_flows
                                    if total_high_level_flows_2 >0:
                                        fraction_of_flows_route_down_indirectly_more_than_one_level[time_epoch_idx][samp_idx]=fraction_of_flows_route_down_indirectly_more_than_one_level[time_epoch_idx][samp_idx]/total_high_level_flows_2

                            # save this for the next run
                            # when deepcopying it changing mem address of flow sources and dest nodes which was bad for next epochs
                            flows_routed_last_run = flows_routed_phase3
                            num_flows_routed_last_run = num_flows_routed_phase3
                            flows_last_run = flows
                            flow_paths_af_last_run = flow_paths_phase3
                            blocked_flows_af_last_run = blocked_flows_phase3
                            waiting_flows_af_last_run = waiting_flows_phase3
                            switches_whos_level_changed_last_run = switches_whos_level_changed_phase3
                            failed_links_af_last_run = failed_links
                            conflict_switches_for_this_flow_af_last_run = conflict_switches_for_this_flow_phase3
                            # show topo phase3 epoch
                            # show_graph(OUTPUT_DIR, g, network_type, run_types[run_type_idx], optimization_problem,
                            #            num_switch_values[num_sw_idx],
                            #            num_flows_for_given_num_sw[num_flows_idx], samp_idx, m_percs[m_perc_index],
                            #            time_epoch_idx, nodesList)

                            if measure_agility and time_epoch_idx >= closest_to_burst_recovery_epoch and time_epoch_idx % relabeling_period == 0:
                                # This is a reboot epoch after the event has occurred or is still occurring
                                # It might recover in the first relabelling or a subsequent one
                                switches_whos_level_changed_recovery_epochs[samp_idx][
                                    time_epoch_idx] = switches_whos_level_changed_phase3
                                flows_affected_recovery_epochs[samp_idx][time_epoch_idx] = (
                                    disruption_perc_flows_booted[run_type_idx][num_sw_idx][num_flows_idx][
                                        samp_idx][m_perc_index][
                                        time_epoch_idx],
                                    disruption_perc_flows_path_changed[run_type_idx][num_sw_idx][num_flows_idx][
                                        samp_idx][m_perc_index][
                                        time_epoch_idx],
                                    disruption_perc_flows_queued[run_type_idx][num_sw_idx][num_flows_idx][
                                        samp_idx][m_perc_index][
                                        time_epoch_idx])

                            plt.close()

                            # Code to run non relabelling on the same topo for agility purposes
                            if measure_agility and time_epoch_idx >= first_event_start:

                                print("********** Running non relabelling ****************")
                                # get network info first (moved here from run_flow_based_heuristic so we can get info on all nodes before trying to fail links; if we fail links before then some nodes wont be in the info lists)
                                if (non_relabelling_flows is not None) or (
                                        non_relabelling_switches_whos_level_changed_last_run is not None):
                                    non_relabelling_flows, non_relabelling_departed_flows, non_relabelling_switches_whos_level_changed_last_run = update_remaining_times(
                                        non_relabelling_flows, non_relabelling_g,
                                        non_relabelling_flow_paths_af_last_run,
                                        non_relabelling_conflict_switches_for_this_flow_af_last_run,
                                        non_relabelling_switches_whos_level_changed_last_run)

                                if non_relabelling_blocked_flows_af_last_run is not None:
                                    non_relabelling_blocked_flows_af_last_run = update_retry_count_on_blocked_flows(
                                        non_relabelling_blocked_flows_af_last_run,
                                        non_relabelling_waiting_flows_af_last_run)
                                    non_relabelling_flows |= non_relabelling_blocked_flows_af_last_run
                                non_relabelling_flows |= gen_flows(
                                    non_relabelling_key_start_id, num_flows_for_given_num_sw[num_flows_idx],
                                    non_relabelling_hosts,
                                    time_epoch_idx)
                                non_relabelling_key_start_id += num_flows_for_given_num_sw[num_flows_idx]

                                non_relabelling_failed_links = non_relabelling_failed_links_af_last_run
                                if link_fails_flag and (time_epoch_idx >= link_fails_start_idx) and (
                                        (time_epoch_idx % link_fail_period) == link_fail_duration):
                                    refresh_failed_links(
                                        non_relabelling_g,
                                        non_relabelling_failed_links_af_last_run)  # put those links back in; can simulate an n-epoch fail with an if check
                                    non_relabelling_failed_links = set()

                                if host_migration_flag and (time_epoch_idx >= host_migration_start_idx) and (
                                        time_epoch_idx % host_migration_period == 0):
                                    if one_event_burst and time_epoch_idx == host_migration_period or not one_event_burst:
                                        # TODO - replicate the same host migration from the relabelling run
                                        replicate_host_migration_event(g, non_relabelling_g)
                                        # change the host locations before collecting info for this epoch
                                        # simulate_change_host_locations(
                                        #     non_relabelling_g, network_type, num_switch_values[num_sw_idx])

                                non_relabelling_switches_pos_dict = {non_relabelling_g['SWITCHES'][i].key: i for i
                                                                     in
                                                                     range(
                                                                         len(non_relabelling_g['SWITCHES']))}
                                non_relabelling_hosts_pos_dict = {non_relabelling_g['HOSTS'][i].key: i for i in
                                                                  range(
                                                                      len(non_relabelling_g['HOSTS']))}

                                non_relabelling_sw_info, non_relabelling_hosts_info = get_switch_degrees_and_neighbor_info(
                                    non_relabelling_g, non_relabelling_switches_pos_dict,
                                    non_relabelling_hosts_pos_dict, non_relabelling_failed_links)

                                # simulate link fails again
                                # failed_links = set() # moved above
                                if link_fails_flag and (time_epoch_idx >= link_fails_start_idx) and (
                                        time_epoch_idx % link_fail_period == 0):
                                    if (
                                            one_event_burst and time_epoch_idx == link_fail_period) or not one_event_burst:
                                        non_relabelling_failed_links = replicate_failed_links(failed_links,
                                                                                              non_relabelling_g)

                                if run_types[run_type_idx] == 'Heuristic':
                                    _, _, _, non_relabelling_flows_routed_phase3, non_relabelling_num_flows_routed_phase3, non_relabelling_num_flows_routed_phase3_by_level, non_relabelling_num_flows_sorted_phase3, _, non_relabelling_flow_paths_phase3, non_relabelling_conflict_switches_for_this_flow_phase3, _, non_relabelling_blocked_flows_phase3, _, non_relabelling_flows_failing_tx_down_delta_phase3, _, non_relabelling_flows_failing_tx_down_num_phase3, non_relabelling_waiting_flows_phase3, \
                                    _, \
                                    _, \
                                    _, \
                                    _, non_relabelling_num_switches_with_level_change_phase3, non_relabelling_switches_whos_level_changed_phase3 = run_sim(
                                        non_relabelling_g, non_relabelling_nodesList, non_relabelling_flows,
                                        network_type,
                                        0, time_epoch_idx, non_relabelling_switches_pos_dict,
                                        non_relabelling_hosts_pos_dict,
                                        non_relabelling_sw_info,
                                        non_relabelling_hosts_info, non_relabelling_failed_links, relabeling_period,
                                        non_relabelling_switches_whos_level_changed_last_run)

                                    non_relabelling_coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                        m_perc_index][
                                        time_epoch_idx] = non_relabelling_num_flows_routed_phase3 / \
                                                          non_relabelling_num_flows_sorted_phase3 * 100

                                    non_relabelling_flows_routed_last_run = non_relabelling_flows_routed_phase3
                                    non_relabelling_num_flows_routed_last_run = non_relabelling_num_flows_routed_phase3
                                    non_relabelling_flows_last_run = non_relabelling_flows
                                    non_relabelling_flow_paths_af_last_run = non_relabelling_flow_paths_phase3
                                    non_relabelling_blocked_flows_af_last_run = non_relabelling_blocked_flows_phase3
                                    non_relabelling_waiting_flows_af_last_run = non_relabelling_waiting_flows_phase3
                                    non_relabelling_switches_whos_level_changed_last_run = non_relabelling_switches_whos_level_changed_phase3
                                    non_relabelling_failed_links_af_last_run = non_relabelling_failed_links
                                    non_relabelling_conflict_switches_for_this_flow_af_last_run = non_relabelling_conflict_switches_for_this_flow_phase3

    for run_type_idx in range(len(run_types)):
        for num_sw_idx in range(len(num_switch_values)):
            for num_flows_idx in range(len(num_flows_for_given_num_sw)):
                # got rid of else case because we can capture one-shot by setting epochs to 1
                for m_perc_index in range(len(m_percs)):
                    for samp_idx in range(num_samples): # print these for each sample so we have the actual data
                        print("Coverage results [topo=%s-type=%s-opt=%d-num_sw_val=%d-nflows=%d-M=%.2f-sample=%d]: " % (network_type, run_types[run_type_idx], optimization_problem, num_switch_values[num_sw_idx], num_flows_for_given_num_sw[num_flows_idx], m_percs[m_perc_index], samp_idx),
                            coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index])

                    ### Plot 1 - get average disruption per epoch ###
                    # account for solver models that may not be solvable and so results have some None entries
                    # NOTE (07/20/2020): NaN coming from epoch 0 since there is no disruption at epoch 0, fix later but the data point shows as 0 anyway
                    fig = plt.figure()
                    plt.rcParams["font.family"] = "Times New Roman"
                    plt.plot([i for i in range(first_event_start - 100, num_epochs)],
                             [np.mean([disruption_perc_flows_booted[run_type_idx][num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] for samp_idx in
                                       range(num_samples) if
                                       disruption_perc_flows_booted[run_type_idx][
                                           num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] is not None]) for ep in
                              range(first_event_start - 100, num_epochs)],
                             marker='s', markersize=0.5, linewidth=1,color='red', label='Booted')
                    """ Path changed is not needed in the plot
                    plt.plot([i for i in range(first_event_start - 100, num_epochs)],
                             [np.mean([disruption_perc_flows_path_changed[run_type_idx][num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] for samp_idx in
                                       range(num_samples) if
                                       disruption_perc_flows_path_changed[
                                           run_type_idx][num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] is not None]) for ep in
                              range(first_event_start - 100, num_epochs)],
                             marker='x', markersize=0.5, linewidth=1, label='Path changed') """
                    plt.plot([i for i in range(first_event_start - 100, num_epochs)],
                             [np.mean([disruption_perc_flows_queued[run_type_idx][num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] for samp_idx in
                                       range(num_samples) if
                                       disruption_perc_flows_queued[run_type_idx][
                                           num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] is not None]) for ep in
                              range(first_event_start - 100, num_epochs)],
                             marker='v', markersize=0.5, linewidth=1,color='green', label='Queued')
                    plt.plot([i for i in range(first_event_start - 100, num_epochs)],
                             [np.mean([coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] for samp_idx in
                                       range(num_samples) if coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] is not None]) for ep in
                              range(first_event_start - 100, num_epochs)],
                             marker='s', markersize=0.5, linewidth=1, color='black', label='Total Coverage')
                    plt.xticks([i for i in range(first_event_start - 100, num_epochs) if i % 500 == 0], [
                        str(i) for i in range(first_event_start - 100, num_epochs) if i % 500 == 0], fontsize=26)
                    plt.yticks(fontsize=26)
                    plt.xlabel('Time (s)', fontsize=12)
                    plt.ylim(0, 100)
                    # plt.ylabel("Disruption/Coverage/TX-downs (%d samples)" % (num_samples), fontsize=12)
                    plt.ylabel("Disruption/Coverage %% (%d samples)" %
                               (num_samples), fontsize=12)
                    plt.grid(True, color='grey', linestyle=':',
                             alpha=0.3, linewidth=0.25)
                    # plt.title("Avg disruption over time, %s [M=%.2f, #sw=%d, #f=%d, warmup=%d]" % (
                    #     str(run_types), m_percs[m_perc_index], num_switch_values[num_sw_idx],
                    #     num_flows_for_given_num_sw[num_flows_idx], num_warmup_epochs), fontsize=12)
                    """ No vlines for relabelling epochs
                    plt.vlines(x=[i for i in range(closest_to_burst_recovery_epoch - 3 * relabeling_period, num_epochs,
                                                   relabeling_period)],
                               color='black', linestyle='--', ymin=0, ymax=100, linewidth=0.5,
                               label='Relabeling epochs') """
                    #plt.legend(fontsize=10, loc='best')
                    plt.tight_layout()
                    if not measure_agility:
                        plt.savefig(
                            '%s/fig-Avg-disruption-over-time-topo=%s-type=%s-opt=%d-num_sw_val=%d-nflows=%d-M=%.2f.pdf' % (
                                OUTPUT_DIR,
                                network_type,
                                run_types[run_type_idx], optimization_problem, num_switch_values[
                                    num_sw_idx],
                                num_flows_for_given_num_sw[num_flows_idx], m_percs[m_perc_index]))
                    else:
                        plt.savefig(
                            '%s/fig-Avg-disruption-over-time-topo=%s-type=%s-opt=%d-num_sw_val=%d-nflows=%d-M=%.2f-event=%s.pdf' % (
                                OUTPUT_DIR,
                                network_type,
                                run_types[run_type_idx], optimization_problem, num_switch_values[
                                    num_sw_idx],
                                num_flows_for_given_num_sw[num_flows_idx], m_percs[m_perc_index],event_type))
                    plt.close()

                    ### Plot 2 - get average coverage per level per epoch ###
                    fig = plt.figure()
                    for lev in range(min_sec_lvl, max_sec_lvl + 1):
                        plt.plot([i for i in range(num_epochs)], [np.mean(
                            [coverages_by_level_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                 ep][lev] for samp_idx in
                             range(num_samples) if
                             coverages_by_level_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                 ep][lev] is not None]) for ep
                            in range(
                                num_epochs)], marker='s', markersize=0.5, linewidth=1,
                                 label='Level=%d, avgcount=%.2f' % (
                                     lev,
                                     np.mean([flow_counts_by_level[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                                  m_perc_index][time_epoch_idx][lev] for
                                              time_epoch_idx in range(num_epochs) for samp_idx in range(num_samples)])))
                    plt.plot([i for i in range(num_epochs)],
                             [np.mean([coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] for samp_idx in
                                       range(num_samples) if coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                       [samp_idx][m_perc_index][ep] is not None]) for ep in
                              range(num_epochs)],
                             marker='s', markersize=0.5, linewidth=1, color='black',
                             label='Total Coverage, avgcount=%.2f' % np.mean(
                                 [len(all_flow_sets[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                          time_epoch_idx]) for
                                  time_epoch_idx in
                                  range(num_epochs) for samp_idx in
                                  range(num_samples)]))  # steady-state number of total flows
                    if (optimization_problem == 1) or (optimization_problem == 2):
                        # only show this for opt1/2 because opt3 is only concerned with tx/route downs. And cant normalize opt3 to [0, 100] because we dont have a maximum number of transfer downs (could be infinity)
                        plt.plot([i for i in range(num_epochs)], [get_obj_val(
                            [all_flow_sets[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][ep]
                             for samp_idx in range(num_samples)],
                            [all_flows_routed_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                 ep] for samp_idx in
                             range(num_samples)], num_samples) for ep in range(num_epochs)],
                                 marker='s', markersize=0.5, linewidth=1, label='Objective val (normalized)')
                    elif optimization_problem == 3:  # This is the objective val in essence
                        plt.plot([i for i in range(num_epochs)],
                                 [np.mean([avg_num_tx_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                               m_perc_index][ep]
                                           for samp_idx in range(num_samples) if
                                           avg_num_tx_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                               m_perc_index][ep] is not None])
                                  for ep in range(num_epochs)], marker='v',
                                 markersize=0.5, linewidth=1, label='TX-downs (Obj=avgcount*tx-downs)')
                        plt.plot([i for i in range(num_epochs)],
                                 [np.mean([avg_num_route_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][
                                               samp_idx][m_perc_index][ep]
                                           for samp_idx in range(num_samples) if
                                           avg_num_route_downs_phase3[run_type_idx][num_sw_idx][num_flows_idx][
                                               samp_idx][m_perc_index][
                                               ep] is not None]) for ep in range(num_epochs)], marker='v',
                                 markersize=0.5, linewidth=1, label='Route-downs')
                    else:
                        print("bad optimization num in plot disruption")
                        exit(1)
                    plt.xticks([i for i in range(num_epochs) if i % 250 == 0], [
                        str(i) for i in range(num_epochs) if i % 250 == 0], fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.xlabel('Time (s)', fontsize=12)
                    plt.ylim(0, 100)
                    plt.ylabel("Coverage (%d samples)" %
                               (num_samples), fontsize=12)
                    plt.grid(True, color='grey', linestyle=':',
                             alpha=0.3, linewidth=0.25)
                    plt.title(
                        "Coverage over time, %s [M=%.2f, #sw=%d, #f=%d, warmup=%d]" % (
                            str(
                                run_types), m_percs[m_perc_index], num_switch_values[num_sw_idx],
                            num_flows_for_given_num_sw[num_flows_idx], num_warmup_epochs), fontsize=12)
                    plt.vlines(x=[i for i in range(num_warmup_epochs, num_epochs, relabeling_period)],
                               color='black', linestyle='--', ymin=0, ymax=100, linewidth=0.5,
                               label='Relabeling epochs')
                    plt.legend(fontsize=12, loc='lower right')
                    plt.tight_layout()
                    plt.savefig(
                        '%s/fig-coverage-by-level-over-time-topo=%s-type=%s-opt=%d-num_sw_val=%d-nflows=%d-M=%.2f.pdf' % (
                            OUTPUT_DIR,
                            network_type,
                            run_types[run_type_idx], optimization_problem, num_switch_values[
                                num_sw_idx],
                            num_flows_for_given_num_sw[num_flows_idx], m_percs[m_perc_index]))
                    plt.close()

                    ### Plot 3 - plot switch level distribution ###
                    fig = plt.figure()
                    for lev in range(min_sec_lvl, max_sec_lvl + 1):
                        # plt.plot([i for i in range(num_epochs)], [np.mean([coverages_by_level_phase3[num_flows_idx][samp_idx][m_perc_index][ep][lev] for samp_idx in range(num_samples) if coverages_by_level_phase3[num_flows_idx][samp_idx][m_perc_index][ep][lev] is not None]) for ep in range(num_epochs)], marker='s', markersize=0.5, linewidth=1, label='Level=%d, avgcount=%.2f' % (lev, np.mean([flow_counts_by_level[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][time_epoch_idx][lev] for time_epoch_idx in range(num_epochs) for samp_idx in range(num_samples)])))
                        plt.plot([i for i in range(num_epochs)], [np.mean(
                            [switch_counts_by_level[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                 ep][lev] for samp_idx in
                             range(num_samples) if
                             switch_counts_by_level[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                 ep][lev] is not None]) * 100.0 /
                                                                  num_switch_values[num_sw_idx]
                                                                  for ep in range(num_epochs)], marker='s',
                                 markersize=0.5,
                                 linewidth=1, label='Level=%d, avgcount=%.2f' % (lev, np.mean(
                                [switch_counts_by_level[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                     m_perc_index][time_epoch_idx][lev] for
                                 time_epoch_idx in range(num_epochs) for samp_idx in range(num_samples)])))
                    plt.xticks([i for i in range(num_epochs) if i % 50 == 0], [
                        str(i) for i in range(num_epochs) if i % 50 == 0], fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.xlabel('Time (s)', fontsize=12)
                    plt.ylim(0, 100)
                    plt.ylabel("Fraction of switches (%d samples)" %
                               (num_samples), fontsize=12)
                    plt.grid(True, color='grey', linestyle=':',
                             alpha=0.3, linewidth=0.25)
                    plt.title("Level distribution over time, %s [M=%.2f, #sw=%d, #f=%d, warmup=%d]" % (
                        str(run_types), m_percs[m_perc_index], num_switch_values[num_sw_idx],
                        num_flows_for_given_num_sw[num_flows_idx], num_warmup_epochs), fontsize=12)
                    plt.vlines(x=[i for i in range(num_warmup_epochs, num_epochs, relabeling_period)],
                               color='black', linestyle='--', ymin=0, ymax=100, linewidth=0.5,
                               label='Relabeling epochs')
                    plt.legend(fontsize=12, loc='upper right')
                    plt.tight_layout()
                    plt.savefig(
                        '%s/fig-level-dist-over-time-topo=%s-type=%s-opt=%d-num_sw_val=%d-nflows=%d-M=%.2f.pdf' % (
                            OUTPUT_DIR,
                            network_type, run_types[
                                run_type_idx],
                            optimization_problem, num_switch_values[
                                num_sw_idx], num_flows_for_given_num_sw[num_flows_idx],
                            m_percs[m_perc_index]))
                    plt.close()

                    ### Plot 4 - security metrics ###
                    if (optimization_problem == 1 or optimization_problem == 3) and (show_sec_metric_flag):
                        # Code to compute average coverage and disruption for an experiment -coverage, disruption averaged across epochs and samples
                        # Assuming that in a given run there is only one fixed M, a single fixed # SWs and fixed # flows
                        avg_flow_coverage_epoch = dict()
                        avg_disruption_perc_flows_path_changed = dict()
                        avg_disruption_perc_flows_queued = dict()
                        avg_disruption_perc_flows_booted = dict()
                        for ep in range(num_epochs):
                            avg_flow_coverage_epoch[ep] = np.mean(
                                [coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][ep]
                                 for samp_idx in
                                 range(num_samples)
                                 if coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                     ep] is not None])
                            avg_disruption_perc_flows_path_changed[ep] = np.mean(
                                [disruption_perc_flows_path_changed[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                     m_perc_index][ep] for samp_idx
                                 in
                                 range(num_samples) if
                                 disruption_perc_flows_path_changed[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                     m_perc_index][ep] is not None])
                            avg_disruption_perc_flows_queued[ep] = np.mean(
                                [disruption_perc_flows_queued[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                     m_perc_index][ep] for samp_idx in
                                 range(num_samples) if
                                 disruption_perc_flows_queued[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                     m_perc_index][ep] is not None])
                            avg_disruption_perc_flows_booted[ep] = np.mean(
                                [disruption_perc_flows_booted[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                     m_perc_index][ep] for samp_idx in
                                 range(num_samples) if
                                 disruption_perc_flows_booted[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][
                                     m_perc_index][ep] is not None])

                        avg_flow_coverage_epoch = np.mean(
                            [avg_flow_coverage_epoch[ep] for ep in range(num_epochs)])
                        avg_disruption_perc_flows_path_changed = np.mean([
                            avg_disruption_perc_flows_path_changed[ep] for ep in range(num_epochs)])
                        avg_disruption_perc_flows_queued = np.mean([
                            avg_disruption_perc_flows_queued[ep] for ep in range(num_epochs)])
                        avg_disruption_perc_flows_booted = np.mean([
                            avg_disruption_perc_flows_booted[ep] for ep in range(num_epochs)])
                        if DEBUG:
                            print("Average coverage per epoch per sample: ",
                                  avg_flow_coverage_epoch)
                            print("Average disruption perc flows path changed per epoch per sample: ",
                                  avg_disruption_perc_flows_path_changed)
                            print("Average disruption perc flows queued per epoch per sample: ",
                                  avg_disruption_perc_flows_queued)
                            print("Average disruption perc flows booted per epoch per sample: ",
                                  avg_disruption_perc_flows_booted)

                        if optimization_problem == 1 or optimization_problem == 3:
                            if DEBUG:
                                print("SECURITY METRIC PLOT")
                                total_flow_downgrade_risk = dict()
                                total_switch_downgrade_risk = dict()
                                for time_epoch_idx in range(num_epochs):
                                    total_flow_downgrade_risk[time_epoch_idx] = dict()
                                    for samp_idx in range(num_samples):
                                        total_flow_downgrade_risk[time_epoch_idx][samp_idx] = 0
                                        for flow in flow_downgrade_risk[time_epoch_idx][samp_idx].keys():
                                            total_flow_downgrade_risk[time_epoch_idx][samp_idx] = \
                                                total_flow_downgrade_risk[time_epoch_idx][samp_idx] + \
                                                flow_downgrade_risk[time_epoch_idx][samp_idx][flow]

                                for time_epoch_idx in range(num_epochs):
                                    total_switch_downgrade_risk[time_epoch_idx] = dict()
                                    for samp_idx in range(num_samples):
                                        total_switch_downgrade_risk[time_epoch_idx][samp_idx] = 0
                                        for sw in switch_downgrade_risk[time_epoch_idx][samp_idx].keys():
                                            total_switch_downgrade_risk[time_epoch_idx][samp_idx] = \
                                                total_switch_downgrade_risk[time_epoch_idx][samp_idx] + \
                                                switch_downgrade_risk[time_epoch_idx][samp_idx][sw]

                                total_flow_downgrade_risk = [np.mean(
                                    [total_flow_downgrade_risk[time_epoch_idx][samp_idx] for samp_idx in
                                     range(num_samples)]) for
                                    time_epoch_idx in
                                    range(num_epochs)]

                                print(total_flow_downgrade_risk)

                                total_switch_downgrade_risk = [np.mean(
                                    [total_switch_downgrade_risk[time_epoch_idx][samp_idx] for samp_idx in
                                     range(num_samples)]) for
                                    time_epoch_idx in
                                    range(num_epochs)]

                                print(total_switch_downgrade_risk)

                            fig = plt.figure()
                            plt.plot([i for i in range(num_epochs)],
                                     total_switch_downgrade_risk,
                                     marker='s', markersize=0.5, linewidth=1, label='total sw downgrade risk')
                            for sw in switch_downgrade_risk[0][0].keys():
                                plt.plot([i for i in range(num_epochs)],
                                         [np.mean(
                                             [switch_downgrade_risk[time_epoch_idx][samp_idx][sw] for samp_idx in
                                              range(num_samples)]) for
                                             time_epoch_idx in
                                             range(num_epochs)]
                                         ,
                                         marker='s', markersize=0.5, linewidth=1, label='sw-%s' % (sw))
                            plt.yticks(fontsize=12)
                            plt.xlabel('Time (s)', fontsize=12)
                            plt.xticks([i for i in range(num_epochs) if i % 10 == 0], [
                                str(i) for i in range(num_epochs) if i % 10 == 0], fontsize=12)
                            # plt.ylim(0, 100)
                            plt.ylabel("Switch  Downgrade Risk  (%d samples)" %
                                       (num_samples), fontsize=12)
                            plt.grid(True, color='grey', linestyle=':',
                                     alpha=0.3, linewidth=0.25)
                            plt.title(
                                " Total Switch Downgrade  Risk over time, %s [M=%.2f, #sw=%d, #f=%d, warmup=%d]" % (
                                    str(
                                        run_types), m_percs[m_perc_index], num_switch_values[num_sw_idx],
                                    num_flows_for_given_num_sw[num_flows_idx], num_warmup_epochs),
                                fontsize=12)
                            plt.vlines(x=[i for i in range(num_warmup_epochs, num_epochs, relabeling_period)],
                                       color='black', linestyle='--', ymin=0, ymax=100, linewidth=0.5,
                                       label='Relabeling epochs')
                            plt.legend(fontsize=3, loc='lower right')
                            plt.tight_layout()
                            plt.savefig(
                                '%s/fig-security-sw-over-time-topo=%s-type=%s-opt=%d-num_sw_val=%d-nflows=%d-M=%.2f.pdf' % (
                                    OUTPUT_DIR,
                                    network_type, str(run_types), optimization_problem, num_switch_values[
                                        num_sw_idx],
                                    num_flows_for_given_num_sw[
                                        num_flows_idx],
                                    m_percs[m_perc_index]))
                            plt.close()

                            fig = plt.figure()
                            plt.plot([i for i in range(num_epochs)],
                                     total_flow_downgrade_risk,
                                     marker='s', markersize=0.5, linewidth=1, label='total fw downgrade risk')
                            plt.yticks(fontsize=12)
                            plt.xlabel('Time (s)', fontsize=12)
                            plt.xticks([i for i in range(num_epochs) if i % 10 == 0], [
                                str(i) for i in range(num_epochs) if i % 10 == 0], fontsize=12)
                            # plt.ylim(0, 100)
                            plt.ylabel("Flow  Downgrade Risk  (%d samples)" %
                                       (num_samples), fontsize=12)
                            plt.grid(True, color='grey', linestyle=':',
                                     alpha=0.3, linewidth=0.25)
                            plt.title(
                                " Total Flow Downgrade Risk, %s [M=%.2f, #sw=%d, #f=%d, warmup=%d]" % (
                                    str(
                                        run_types), m_percs[m_perc_index], num_switch_values[num_sw_idx],
                                    num_flows_for_given_num_sw[num_flows_idx], num_warmup_epochs),
                                fontsize=12)
                            plt.vlines(x=[i for i in range(num_warmup_epochs, num_epochs, relabeling_period)],
                                       color='black', linestyle='--', ymin=0, ymax=100, linewidth=0.5,
                                       label='Relabeling epochs')
                            plt.legend(fontsize=4, loc='lower right')
                            plt.tight_layout()
                            plt.savefig(
                                '%s/fig-security-flows-over-time-topo=%s-type=%s-opt=%d-num_sw_val=%d-nflows=%d-M=%.2f.pdf' % (
                                    OUTPUT_DIR,
                                    network_type, str(run_types), optimization_problem, num_switch_values[
                                        num_sw_idx],
                                    num_flows_for_given_num_sw[
                                        num_flows_idx], m_percs[m_perc_index]))
                            plt.close(fig)

    if not measure_agility:
        ## Plot 5 - plot exec time ###
        # (only need 1 of these so moved code outside of loop above)
        # exec time vs # flows
        fig = plt.figure()
        for run_type_idx in range(len(run_types)):
            for m_perc_index in range(len(m_percs)):
                for num_sw_idx in range(len(num_switch_values)):
                    plt.plot([num_flows_for_given_num_sw[num_flows_idx] for num_flows_idx in
                              range(len(num_flows_for_given_num_sw))], [np.mean(
                        [running_times[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][ep] for samp_idx
                         in
                         range(num_samples) for ep in range(
                            num_epochs) if
                         running_times[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                             ep] is not None])
                                 for num_flows_idx in
                                 range(len(num_flows_for_given_num_sw))], marker='x',
                             markersize=5, linewidth=0.5, label='%s, M=%.2f, #sw=%d' % (
                            run_types[run_type_idx], m_percs[m_perc_index], num_switch_values[num_sw_idx]))
        plt.xticks(num_flows_for_given_num_sw, [
            str(f) for f in num_flows_for_given_num_sw], fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Number of flows', fontsize=12)
        plt.ylim(0, 50)
        plt.ylabel("Exec time (s)", fontsize=12)
        plt.grid(True, color='grey', linestyle=':',
                 alpha=0.3, linewidth=0.25)
        plt.title("Exec time vs. Number of flows [M=%.2f, warmup=%d]" % (
            m_percs[m_perc_index], num_warmup_epochs), fontsize=12)
        plt.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.savefig(
            '%s/fig-exectime-vs-numflows-topo=%s-type=%s-opt=%d-num_flows=%s-num_sw_val=%s-M=%.2f.pdf' % (OUTPUT_DIR,
                                                                                                          network_type,
                                                                                                          str(
                                                                                                              run_types),
                                                                                                          optimization_problem,
                                                                                                          str(
                                                                                                              num_flows_for_given_num_sw),
                                                                                                          str(
                                                                                                              num_switch_values),
                                                                                                          m_percs[
                                                                                                              m_perc_index]))
        plt.close()

        # exec time vs # switches
        fig = plt.figure()
        for run_type_idx in range(len(run_types)):
            for m_perc_index in range(len(m_percs)):
                for num_flows_idx in range(len(num_flows_for_given_num_sw)):
                    plt.plot([num_switch_values[num_sw_idx] for num_sw_idx in range(len(num_switch_values))], [np.mean(
                        [running_times[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][ep] for samp_idx
                         in
                         range(num_samples) for ep in range(
                            num_epochs) if
                         running_times[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                             ep] is not None])
                        for
                        num_sw_idx in
                        range(len(
                            num_switch_values))],
                             marker='x', markersize=5, linewidth=0.5, label='%s, M=%.2f, #f=%d' % (
                            run_types[run_type_idx], m_percs[m_perc_index], num_flows_for_given_num_sw[num_flows_idx]))
        plt.xticks(num_switch_values, [
            str(s) for s in num_switch_values], fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Number of switches', fontsize=12)
        plt.ylim(0, 50)
        plt.ylabel("Exec time (s)", fontsize=12)
        plt.grid(True, color='grey', linestyle=':',
                 alpha=0.3, linewidth=0.25)
        plt.title("Exec time vs. Number of switches [M=%.2f, warmup=%d]" % (
            m_percs[m_perc_index], num_warmup_epochs), fontsize=12)
        plt.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.savefig(
            '%s/fig-exectime-vs-numsw-topo=%s-type=%s-opt=%d-num_flows=%s-num_sw_val=%s-M=%.2f.pdf' % (OUTPUT_DIR,
                                                                                                       network_type,
                                                                                                       str(
                                                                                                           run_types),
                                                                                                       optimization_problem,
                                                                                                       str(
                                                                                                           num_flows_for_given_num_sw),
                                                                                                       str(
                                                                                                           num_switch_values),
                                                                                                       m_percs[
                                                                                                           m_perc_index]))
        plt.close()

        ### Plot 6 - coverage (not by level) over time ###
        fig = plt.figure()  # put all on same plot
        plt.rcParams["font.family"] = "Times New Roman"
        for num_sw_idx in range(len(num_switch_values)):
            for num_flows_idx in range(len(num_flows_for_given_num_sw)):
                for m_perc_index in range(len(m_percs)):
                    for run_type_idx in range(len(run_types)):
                        line_sty_obj = '-'
                        line_color_obj = 'blue' if run_type_idx == 0 else 'orange'
                        plt.plot([i for i in range(num_epochs)], [get_obj_val(
                            [all_flow_sets[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][ep]
                             for samp_idx in range(num_samples)],
                            [all_flows_routed_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][
                                 ep] for samp_idx in
                             range(num_samples)], num_samples) / 100.0 for ep in range(num_epochs)],
                                 linewidth=0.5, color=line_color_obj, linestyle=line_sty_obj,
                                 label='%s, Objective val (normalized)' % run_types[run_type_idx])

                        # line_sty = '-.' if run_types[run_type_idx] == 'Heuristic' else '-'
                        # line_color = 'black' if m_perc_index == 0 else 'red'
                        line_sty = '-'
                        line_color = 'black' if run_type_idx == 0 else 'red'
                        plt.plot([i for i in range(num_epochs)],
                                 [np.mean([coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                           [samp_idx][m_perc_index][ep] for samp_idx in
                                           range(num_samples) if
                                           coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                           [samp_idx][m_perc_index][ep] is not None]) / 100.0 for ep in
                                  range(num_epochs)], linewidth=0.5, color=line_color, linestyle=line_sty,
                                 label='%s, M=%.2f' % (run_types[run_type_idx], m_percs[m_perc_index]))

        plt.xticks([i for i in range(num_epochs) if i % 500 == 0], [
            str(i) for i in range(num_epochs) if i % 500 == 0], fontsize=26)
        plt.yticks(fontsize=26)
        # plt.tick_params(direction='in')
        plt.xlabel('Time (s)', fontsize=26)
        plt.ylabel("Coverage (%)", fontsize=26)
        plt.ylim(0, 1)
        plt.grid(True, color='grey', linestyle=':',
                 alpha=0.3, linewidth=0.25)
        # plt.title("Coverage over time", fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(
            '%s/fig-coverage-over-time-topo=%s-type=%s-opt=%d-num_flows=%s-num_sw_val=%s-M=%.2f.pdf' % (OUTPUT_DIR,
                                                                                                        network_type,
                                                                                                        str(
                                                                                                            run_types),
                                                                                                        optimization_problem,
                                                                                                        str(
                                                                                                            num_flows_for_given_num_sw),
                                                                                                        str(
                                                                                                            num_switch_values),
                                                                                                        m_percs[
                                                                                                            m_perc_index]))
        plt.close()

    #### NEW PLOT 7 - AGILITY ###
    if measure_agility:
        for run_type_idx in range(len(run_types)):
            for num_sw_idx in range(len(num_switch_values)):
                for num_flows_idx in range(len(num_flows_for_given_num_sw)):
                    for m_perc_index in range(len(m_percs)):
                        high = [[0 for _ in range(num_event_bursts)] for _ in range(num_samples)]
                        for sample_id in range(0, num_samples):
                            # Take the steady state coverage as the average (Ensure the event occurs after steady state)
                            time_to_adapt[run_type_idx][num_sw_idx][num_flows_idx][m_perc_index][
                                sample_id] = [0 for _ in range(num_event_bursts)]
                            print("Sample - ", sample_id, "Event type -", event_type, " # Bursts", num_event_bursts,
                                  "First event start -",
                                  first_event_start)
                            for event_burst_id in range(num_event_bursts):
                                event_start = first_event_start + event_burst_id * event_period
                                event_end = min(event_start + event_duration, num_epochs)
                                print("Event id - ", event_burst_id, " start ", event_start, " end ", event_end)
                                low = num_epochs
                                adapt_time = 0
                                seen_low = False
                                seen_one_high = False
                                # Take last 200 epochs average (steady state)
                                avg_flow_coverage_epoch = np.mean(
                                    coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                    [sample_id][m_perc_index][
                                    event_start - 200:event_start - 1])
                                print("AVG :", avg_flow_coverage_epoch)

                                for epoch_idx in range(event_start, event_end):
                                    if \
                                            coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][sample_id][
                                                m_perc_index][
                                                epoch_idx] < avg_flow_coverage_epoch:
                                        low = min(low, epoch_idx)
                                        seen_low = True
                                    elif seen_low and \
                                            coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][sample_id][
                                                m_perc_index][
                                                epoch_idx] >= avg_flow_coverage_epoch:
                                        print("TTA - ", epoch_idx, low)
                                        print("Coverage here -",
                                              coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][sample_id][
                                                  m_perc_index][
                                                  epoch_idx])
                                        adapt_time = adapt_time + epoch_idx - low
                                        low = num_epochs
                                        seen_low = False
                                        seen_one_high = True
                                        high[sample_id][event_burst_id] = epoch_idx
                                        if epoch_idx % relabeling_period == reboot_time:
                                            print("# SW's changed - ",
                                                  len(switches_whos_level_changed_recovery_epochs[sample_id][
                                                          epoch_idx - reboot_time]))
                                            print("Disruption (Booted %, path changed %, queued %) - ",
                                                  flows_affected_recovery_epochs[samp_idx][epoch_idx - reboot_time])
                                    elif coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][sample_id][
                                        m_perc_index][
                                        epoch_idx] >= avg_flow_coverage_epoch:
                                        seen_one_high = True

                                print('Total adapt time: ', adapt_time)
                                time_to_adapt[run_type_idx][num_sw_idx][num_flows_idx][m_perc_index][
                                    sample_id][event_burst_id] = adapt_time
                                if high[sample_id][event_burst_id] == 0:
                                    # It didn't reach steady state
                                    if not seen_one_high:
                                        high[sample_id][event_burst_id] = event_end
                                    else:
                                        high[sample_id][event_burst_id] = event_start + 1
                                print("Sample ", sample_id, " event burst id", event_burst_id, " high -",
                                      high[sample_id][event_burst_id])
                            # time_to_adapt[run_type_idx][num_sw_idx][num_flows_idx][m_perc_index][
                            #     samp_idx] = np.mean(
                            #     time_to_adapt[run_type_idx][num_sw_idx][num_flows_idx][m_perc_index][
                            #         samp_idx])

                        # time_to_adapt[run_type_idx][num_sw_idx][num_flows_idx][m_perc_index] = np.mean(
                        #     time_to_adapt[run_type_idx][num_sw_idx][num_flows_idx][m_perc_index])
                        # print("For M %", m_percs[m_perc_index], "  AVG TIME TO ADAPT: ",
                        #       time_to_adapt[run_type_idx][num_sw_idx][num_flows_idx][m_perc_index])
                        high = [np.mean([high[sample_id][event_burst_id] for sample_id in range(num_samples)]) for
                                event_burst_id in range(num_event_bursts)]
                        print("HIGH", high)

        # Plot the agility
        print("Avg time to adapt: ",
              [time_to_adapt[run_type_idx][num_sw_idx][num_flows_idx][m_perc_index] for m_perc_index in
               range(len(m_percs))])
        for run_type_idx in range(len(run_types)):
            for num_sw_idx in range(len(num_switch_values)):
                for num_flows_idx in range(len(num_flows_for_given_num_sw)):
                    fig = plt.figure()
                    plt.rcParams["font.family"] = "Times New Roman"
                    for m_perc_index in range(len(m_percs)):
                        line_sty = '-'
                        line_color = 'black'
                        # non_relabelling_coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][first_event_start-100:first_event_start] = coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx][samp_idx][m_perc_index][first_event_start-100:first_event_start]
                        plt.plot([i for i in range(first_event_start - 100, num_epochs)],
                                 [np.mean(
                                     [non_relabelling_coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                      [samp_idx][m_perc_index][ep] for samp_idx in
                                      range(num_samples) if
                                      non_relabelling_coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                      [samp_idx][m_perc_index][ep] is not None]) / 100.0 for ep
                                  in
                                  range(first_event_start - 100, num_epochs)], linewidth=0.5, color=line_color,
                                 linestyle=line_sty,
                                 label='%s' % "No relabelling")

                        line_sty = '-'
                        line_color = 'blue'
                        plt.plot([i for i in range(first_event_start - 100, num_epochs)],
                                 [np.mean([coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                           [samp_idx][m_perc_index][ep] for samp_idx in
                                           range(num_samples) if
                                           coverages_phase3[run_type_idx][num_sw_idx][num_flows_idx]
                                           [samp_idx][m_perc_index][ep] is not None]) / 100.0 for ep
                                  in
                                  range(first_event_start - 100, num_epochs)], linewidth=0.5, color=line_color,
                                 linestyle=line_sty,
                                 label='%s, M=%.2f' % (
                                     run_types[run_type_idx], m_percs[m_perc_index]))

                        for burst_id in range(num_event_bursts):
                            event_start = first_event_start + burst_id * event_period
                            event_end = event_start+ event_duration
                            print("Highlighted region - ", event_start, event_end)
                            plt.axvspan(event_start, event_end, color='red', alpha=0.3)
                            # plt.axvspan(event_start, high[burst_id], color='red', alpha=0.3)

                    plt.xticks([i for i in range(first_event_start - 100, num_epochs) if i % 500 == 0],
                               [str(i) for i in range(first_event_start - 100, num_epochs) if i % 500 == 0],
                               fontsize=26)
                    plt.yticks(fontsize=26)
                    plt.tick_params(direction='in')
                    # plt.xlabel('M (%)', fontsize=26)
                    plt.xlabel('Time (s)', fontsize=26)
                    plt.ylabel("Coverage (%) ", fontsize=26)
                    plt.ylim(0, 1)
                    plt.grid(True, color='grey', linestyle=':',
                             alpha=0.6, linewidth=0.25)
                    # plt.title("Coverage over time", fontsize=12)
                    plt.legend(fontsize=24, loc='best')
                    plt.tight_layout()
                    plt.savefig(
                        '%s/fig-agility-over-time-topo=%s-type=%s-opt=%d-num_flows=%s-num_sw_val=%s-M=%.2f-event=%s.pdf' % (
                            OUTPUT_DIR,
                            network_type,
                            str(
                                run_types),
                            optimization_problem,
                            str(
                                num_flows_for_given_num_sw),
                            str(
                                num_switch_values),
                            m_percs[
                                m_perc_index],
                            event_type))
                    plt.close(fig)

    #### NEW - Print security stats #######
    if show_sec_metric_flag:
        print("*****************************************************************************************************************")
        print("AVG fraction of high levels flows routed through switches that may route down  (sw.key: # flows)",{sw.key:np.mean(
        [np.mean([fraction_of_high_flows_per_switch_may_route_down[time_epoch_idx][samp_idx][sw.key]  for samp_idx in range(num_samples)]) for
            time_epoch_idx in
            range(num_epochs)]) for sw in g['SWITCHES']})

        print("AVG fraction of high levels flows routed that may route down through switches that may route down  (sw.key: # flows)",
              {sw.key: np.mean(
                  [np.mean(
                      [fraction_of_high_flows__may_route_down_per_switch_may_route_down[time_epoch_idx][samp_idx][sw.key] for samp_idx
                       in range(num_samples)]) for
                   time_epoch_idx in
                   range(num_epochs)]) for sw in g['SWITCHES']})


        print("AVG fraction of high levels flows routed through a switch that may route down  (sw.key: # flows)",np.mean([np.mean(
        [np.mean([fraction_of_high_flows_per_switch_may_route_down[time_epoch_idx][samp_idx][sw.key]  for samp_idx in range(num_samples)]) for
            time_epoch_idx in
            range(num_epochs)]) for sw in g['SWITCHES']]))


        print("AVG fraction of high levels flows routed that may route down through switches that may route down  (sw.key: # flows)",np.mean([np.mean(
        [np.mean([fraction_of_high_flows__may_route_down_per_switch_may_route_down[time_epoch_idx][samp_idx][sw.key]  for samp_idx in range(num_samples)]) for
            time_epoch_idx in
            range(num_epochs)]) for sw in g['SWITCHES']]))

        print("AVG fraction of flows that use switches that may route down ", np.mean([np.mean([fraction_of_flows_use_switch_route_down[time_epoch_idx][samp_idx] for samp_idx in range(num_samples)  ]) for time_epoch_idx in range(num_epochs)]))

        print("AVG fraction of flows that may be route down more than one level (indirectly) ", np.mean([np.mean([fraction_of_flows_route_down_indirectly_more_than_one_level[time_epoch_idx][samp_idx] for samp_idx in range(num_samples)  ]) for time_epoch_idx in range(num_epochs)]))


    return


if __name__ == '__main__':
    random.seed(time.time())
    run_exp()
    print("[Done]")
