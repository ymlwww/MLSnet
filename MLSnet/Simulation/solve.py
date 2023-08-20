#!/usr/bin/env python3.7
#
# solve.py
#
# Description   : Build and execute optimization model with gurobi solver python api.
# Created by    :
# Date          : November 2019
# Last Modified : June 2020

# https://stackoverflow.com/questions/50366433/inverted-indicator-constraint-in-gurobipy

### Imports ###
from gurobipy import *
from common import extractLinkMatrix, B, num_levels, min_sec_lvl, max_sec_lvl, delta_j, delta_c_j, optimization_problem, \
    flow_weight, DEBUG, get_num_route_downs, get_info_type, num_warmup_epochs, relabeling_period, reboot_time, switch_cap
import time
import numpy as np
from copy import deepcopy
from sim import findFeasiblePaths

# Global BIG M - Note for OPT 2 BIG M >= (max-sec-level - min-sec-level)
BIG_M = 10


# USING TRANSPOSE EQUATIONS TO SUPPLEMENT BIG M CONSTRAINTS
### Functions ###


def _solve_op_one(g, nodesList, flows, M, time_epoch, failed_links):

    ## Preprocessing of sorts
    failed_links = [l[0] for l in failed_links]
    ###### Step 1 - create model ######
    m = Model('Relabeling')
    m.reset(0)
    m.remove(m.getVars())
    m.remove(m.getGenConstrs())

    ###### Step 2 - add variables ######
    x_i = dict()  # x_i - starting level of switch i
    # was doing nodesList before for use in x_j_kl but being more explicit now and just setting the constraints directly with the node levels (as constants) since they shouldnt change
    for sw in g['SWITCHES']:
        x_i[sw.key] = sw.level

    # need this for x_i_pr
    delta_x_i = dict()
    for sw in g['SWITCHES']:
        if time_epoch < num_warmup_epochs:
            delta_x_i[sw.key] = m.addVar(
                lb=0, ub=0, vtype=GRB.INTEGER, name='delta_x_i-%s' % (sw.key))
        else:
            delta_x_i[sw.key] = m.addVar(lb=((-1) * x_i[sw.key]) + 1, ub=num_levels -
                                         x_i[sw.key], vtype=GRB.INTEGER,
                                         name='delta_x_i-%s' % (sw.key))

    # x_i_pr (prime) - *new* level of switch
    x_i_pr = dict()
    for sw in g['SWITCHES']:
        x_i_pr[sw.key] = m.addVar(
            lb=min_sec_lvl, ub=max_sec_lvl, vtype=GRB.INTEGER, name='x_i_pr-%s' % (sw.key))

    # already filters out link violations (e.g., switch->source is prohibited); edit (5/26): not anymore
    i_kl = extractLinkMatrix(g, nodesList)

    # feasible links
    x_j_kl = dict()
    for j in flows:
        for k in [j.source] + g['SWITCHES']:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS'].keys():
                    x_j_kl[(j.key, k.key, l.key)] = m.addVar(
                        vtype=GRB.BINARY, name='x_j_kl-%s_%s_%s' % (j.key, k.key, l.key))

    xjkl_link_exists_indicator = dict()
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS'].keys():
                xjkl_link_exists_indicator[(k.key, l.key)] = m.addVar(
                    vtype=GRB.BINARY, name='xjkl_link_exists_indicator-%s_%s' % (k.key, l.key))

    xjkl_level_indicator = dict()
    for j in flows:
        for k in [j.source] + g['SWITCHES']:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS'].keys():
                    xjkl_level_indicator[(j.key, k.key, l.key)] = m.addVar(
                        vtype=GRB.BINARY, name='xjkl_level_indicator-%s-%s-%s' % (j.key, k.key, l.key))

    xjkl_transfer_down_ok_indicator = dict()
    xjkl_min_level = dict()
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS'].keys():
                # add the transfer down OK indicator variable
                xjkl_transfer_down_ok_indicator[(k.key, l.key)] = m.addVar(
                    vtype=GRB.BINARY, name='xjkl_transfer_down_ok_indicator-%s_%s' % (k.key, l.key))

                # also add the min level variable
                xjkl_min_level[(k.key, l.key)] = m.addVar(
                    vtype=GRB.INTEGER, name='xjkl_min_level-%s_%s' % (k.key, l.key))

    # for constraint 8, keep track of which links have a transfer down
    xjkl_transfer_down_indicator = dict()
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS'].keys():
                xjkl_transfer_down_indicator[(k.key, l.key)] = m.addVar(
                    vtype=GRB.BINARY, name='xjkl_transfer_down_indicator-%s_%s' % (k.key, l.key))

    i_delta = dict()
    for sw in g['SWITCHES']:
        i_delta[sw.key] = m.addVar(
            vtype=GRB.BINARY, name='i_delta-%s' % (sw.key))

    y = dict()  # links are source-switch, switch-switch, switch-dest (see bottom of slide)
    for j in flows:
        for k in [j.source] + g['SWITCHES']:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS']:
                    y[(j.key, k.key, l.key)] = m.addVar(
                        vtype=GRB.BINARY, name='y-%s_%s_%s' % (j.key, k.key, l.key))

    alpha = dict()
    for j in flows:
        alpha[j.key] = m.addVar(vtype=GRB.BINARY, name='alpha_%s' % (j.key))

    # update model with new vars
    m.update()

    ###### Step 3 - set objective ######
    m.setObjective(
        quicksum(j.demand * alpha[j.key] * flow_weight(j.level) for j in flows), GRB.MAXIMIZE)

    ###### Step 4 - add constraints ######

    # General variable constraints
    for sw in g['SWITCHES']:  # enforce the new level to be the starting level plus some delta
        m.addConstr(x_i_pr[sw.key] == (
            x_i[sw.key] + delta_x_i[sw.key]), 'constr_x_i_pr-%s' % (sw.key))

    # enforce new x_j_kl
    link_exists_indicators_added = set()
    transfer_down_ok_indicators_added = set()
    min_link_level_vars_added = set()
    fixed_host_levels = {h.key: h.level for h in g['HOSTS']}
    for j in flows:
        for k in nodesList:
            for l in nodesList:
                if (j.key, k.key, l.key) in x_j_kl.keys():
                    # diff flows can reuse this indicator
                    if (k.key, l.key) not in link_exists_indicators_added:
                        if (k.key, l.key) not in failed_links:
                            # force this indicator to be the correct value from link matrix
                            m.addConstr(xjkl_link_exists_indicator[(
                                k.key, l.key)] == int(i_kl[k.key][l.key]),
                                'constr_xjkl_link_exists_indicator-%s_%s' % (k.key, l.key))
                        else:
                            # force this indicator to be 0 as the link has failed
                            m.addConstr(xjkl_link_exists_indicator[(
                                k.key, l.key)] == int(0),
                                'constr_xjkl_link_exists_indicator-%s_%s' % (k.key, l.key))
                        link_exists_indicators_added.add((k.key, l.key))
                    lvl_check_dict_k = get_info_type(
                        k.key, x_i_pr, fixed_host_levels, fstr=False)
                    lvl_check_dict_l = get_info_type(
                        l.key, x_i_pr, fixed_host_levels, fstr=False)

                    # reuse for flows that may travel along this link
                    if (k.key, l.key) not in min_link_level_vars_added:
                        # Need to check if (SW,SW) or (Src,SW) or (SW,Dst) here (Maybe we can add this to the common function)
                        if k in g['SWITCHES'] and l in g['HOSTS']:
                            m.addConstr(xjkl_min_level[(k.key, l.key)] ==
                                        lvl_check_dict_k[k.key], name='constr_xjkl_min_level-%s_%s' % (k.key, l.key))
                        elif k in g['HOSTS'] and l in g['SWITCHES']:
                            m.addConstr(xjkl_min_level[(k.key, l.key)] ==
                                        lvl_check_dict_l[l.key], name='constr_xjkl_min_level-%s_%s' % (k.key, l.key))
                        else:
                            m.addConstr(xjkl_min_level[(k.key, l.key)] == min_(
                                lvl_check_dict_k[k.key], lvl_check_dict_l[l.key]),
                                name='constr_xjkl_min_level-%s_%s' % (k.key, l.key))
                        min_link_level_vars_added.add((k.key, l.key))

                    # Sets/forces the indicator constraint to be 1 if min level >= flow level
                    m.addConstr(BIG_M * xjkl_level_indicator[(j.key, k.key, l.key)] >= (
                        1 + xjkl_min_level[(k.key, l.key)] - j.level))

                    # Forces it to be zero when flow level > min level
                    m.addConstr(BIG_M * (1 - xjkl_level_indicator[(j.key, k.key, l.key)]) >= (
                        j.level - xjkl_min_level[(k.key, l.key)]))

                    #  Prevents false positives
                    m.addConstr((xjkl_level_indicator[(j.key, k.key, l.key)] == 1) >> (
                        j.level <= xjkl_min_level[(k.key, l.key)]),
                        'constr_xjkl_level_indicator-%s_%s_%s' % (j.key, k.key, l.key))

                    #  Transpose - Alternative constraint to big M - Using both this and big M will not affect correctness
                    m.addConstr((xjkl_level_indicator[(j.key, k.key, l.key)] == 0) >> (
                        j.level - 1 >= xjkl_min_level[(k.key, l.key)]),
                        'constr_xjkl_level_indicator-%s_%s_%s' % (j.key, k.key, l.key))

                    # diff flows can also reuse this indicator
                    if (k.key, l.key) not in transfer_down_ok_indicators_added:
                        # Sets/forces the transfer down indicator to be 1
                        m.addConstr((BIG_M * xjkl_transfer_down_ok_indicator[(k.key, l.key)]) >= (
                            1 + delta_j - (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key])))

                        # Sets/forces the transfer down indicator to be 0
                        m.addConstr((BIG_M * (1 - xjkl_transfer_down_ok_indicator[(k.key, l.key)])) >= (
                            (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key]) - delta_j))

                        #  Prevents false positives
                        m.addConstr((xjkl_transfer_down_ok_indicator[(k.key, l.key)] == 1) >> (
                            (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key]) <= delta_j),
                            'constr_xjkl_transfer_down_ok_indicator-%s_%s' % (k.key, l.key))

                        #  Transpose - Alternative constraint to big M - Using both this and big M will not affect correctness
                        m.addConstr((xjkl_transfer_down_ok_indicator[(k.key, l.key)] == 0) >> (
                            (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key] - 1) >= delta_j),
                            'constr_xjkl_transfer_down_ok_indicator-%s_%s' % (k.key, l.key))

                        transfer_down_ok_indicators_added.add(
                            (k.key, l.key))

                    m.addConstr(x_j_kl[(j.key, k.key, l.key)] == and_(xjkl_link_exists_indicator[(
                        k.key, l.key)], xjkl_level_indicator[(j.key, k.key, l.key)],
                        xjkl_transfer_down_ok_indicator[
                        (k.key, l.key)]),
                        'constr_xjkl-%s_%s_%s' % (j.key, k.key, l.key))

    # enforce i_delta
    # >/< not supported only >=/<= and == (see gurobi constants)
    abs_delta_x = dict()
    for sw in g['SWITCHES']:
        abs_delta_x[sw.key] = m.addVar(
            vtype=GRB.INTEGER, name='abs_delta_x-%s' % sw.key)
        m.addConstr(abs_delta_x[sw.key] == abs_(delta_x_i[sw.key]))

        # Sets/forces i_delta[sw] to be 1 if delta_x_i[sw] != 0
        m.addConstr((BIG_M * i_delta[sw.key]) >= abs_delta_x[sw.key])

        # Sets/forces i_delta[sw] to be 0 if delta_x_i[sw] = 0
        m.addConstr((BIG_M * (1 - i_delta[sw.key]))
                    >= (1 - abs_delta_x[sw.key]))

        m.addConstr((i_delta[sw.key] == 0) >> (
            delta_x_i[sw.key] == 0), 'constr_i_delta-%s' % sw.key)
    # TODO (5/26): maybe add another constraint the other way around?
    # (6/2) : Done these three additional constraints ensure that i_delta is 1(true) when apt

    # Constraint 1
    for j in flows:
        for k in g['SWITCHES'] + [j.source]:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS']:
                    m.addConstr(y[(j.key, k.key, l.key)] <= x_j_kl[(
                        j.key, k.key, l.key)], 'constr1-%s_%s_%s' % (j.key, k.key, l.key))

    # Constraint 2 - (In flow)
    for j in flows:
        m.addConstr(
            quicksum(y[(j.key, j.source.key, k.key)] for k in nodesList if (
                j.key, j.source.key, k.key) in y.keys())
            == alpha[j.key], 'constr2-%s' % (j.key))

    # Constraint 3 - (Out flow)
    for j in flows:
        m.addConstr(
            quicksum(y[(j.key, k.key, j.dest.key)]
                     for k in nodesList if (j.key, k.key, j.dest.key) in y.keys())
            == alpha[j.key], 'constr3-%s' % (j.key))

    # Constraint 4 (Conservation)
    for j in flows:
        for l in g['SWITCHES']:
            m.addConstr(quicksum(y[(j.key, k.key, l.key)] for k in nodesList if
                                 (j.key, k.key, l.key) in y.keys()) == quicksum(
                y[(j.key, l.key, l2.key)] for l2 in nodesList if (j.key, l.key, l2.key) in y.keys()),
                'constr4-%s_%s' % (j.key, l.key))

    # Constraint 5 - No loops
    for j in flows:
        for l in g['SWITCHES']:
            m.addConstr(quicksum(y[(j.key, k.key, l.key)] for k in nodesList if (
                j.key, k.key, l.key) in y.keys()) <= 1, 'constr5-%s-%s-%s' % (j.key, k.key, l.key))

    # Constraint 6 (sum of demands for all flows entering switch i (from anywhere) is less than or equal to switch capacity)
    # Note: also why is k in J U S? Does it mean I U S? Yes, I think so too.
    for sw in g['SWITCHES']:
        # Note: inner loop is to add all the flows coming into i on link k, then outer loop is to add all the sums across all incoming links
        # K should be in I U S, but here k can be dest as well.
        # R: here using g["SOURCES"]+g["SWITCHES"] for k is okay because there is no Y for invalid sources wrt a given flow (I think)
        m.addConstr(
            quicksum(quicksum(j.demand * y[(j.key, k.key, sw.key)] for j in flows if (j.key, k.key, sw.key) in y.keys())
                     for k in nodesList) <= sw.capacity, 'constr6-%s' % (sw.key))

    # Constraint 7 (sum of flow demands using this link do not exceed link capacity)
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                m.addConstr(
                    quicksum(j.demand * y[(j.key, k.key, l.key)] for j in flows if (j.key, k.key, l.key) in y.keys()) <=
                    g['LINKS'][(k.key, l.key)].capacity,
                    'constr7-%s_%s' % (k.key, l.key))

    # Constraint 8 (limit the number of transfer downs by B)
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                lvl_check_dict_k = get_info_type(
                    k.key, x_i_pr, fixed_host_levels, fstr=False)
                lvl_check_dict_l = get_info_type(
                    l.key, x_i_pr, fixed_host_levels, fstr=False)
                m.addConstr((BIG_M * xjkl_transfer_down_indicator[(k.key, l.key)]) >= (
                    (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key])),
                    'constr_xjkl_transfer_down_indicator-%s-%s' % (k.key, l.key))

                m.addConstr((BIG_M * (1 - xjkl_transfer_down_indicator[(k.key, l.key)])) >= (
                    (1 + lvl_check_dict_l[l.key] - lvl_check_dict_k[k.key])),
                    'constr_xjkl_transfer_down_indicator-%s-%s' % (k.key, l.key))

                m.addConstr((xjkl_transfer_down_indicator[(k.key, l.key)] == 0) >> (
                    (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key]) <= 0),
                    'constr_xjkl_transfer_down_indicator-%s-%s' % (k.key, l.key))

    for j in flows:
        m.addConstr(quicksum(quicksum(xjkl_transfer_down_indicator[(k.key, l.key)] * y[(
            j.key, k.key, l.key)] for l in nodesList if (j.key, k.key, l.key) in y.keys()) for k in nodesList) <= B,
            'constr8-%s' % (j.key))

    # Constraint 9 (limit the number of switches whose level changes)
    m.addConstr(quicksum(i_delta[sw.key]
                         for sw in g['SWITCHES']) <= M, 'constr9')

    # m.write("relmodel.lp")
    return m, y, alpha, x_i_pr, x_i, M, x_j_kl, xjkl_link_exists_indicator, xjkl_level_indicator, xjkl_transfer_down_ok_indicator, xjkl_min_level


def _solve_op_two(g, nodesList, flows, M, time_epoch, failed_links):
    """     Optimization Problem 2 - Routing only through switches of the same level
    No routing up, no routing down.
    Changes (differences from OP 1)
      . Same level indicator
      . X_j_k_l formula
      . No constraint 8 as there are no transfer downs                          """

    ## Preprocessing of sorts
    failed_links = [l[0] for l in failed_links]

    ###### Step 1 - create model ######
    m = Model('Relabeling')
    m.reset(0)
    m.remove(m.getVars())
    m.remove(m.getGenConstrs())

    ###### Step 2 - add variables ######
    x_i = dict()  # x_i - starting level of switch i
    for i in g['SWITCHES']:
        x_i[i.key] = i.level

    # Need this for x_i_pr
    delta_x_i = dict()
    for i in g['SWITCHES']:
        # if time_epoch < num_warmup_epochs:
        #     delta_x_i[i.key] = m.addVar(
        #         lb=0, ub=0, vtype=GRB.INTEGER, name='delta_x_i-%s' % (i.key))
        # else:
        delta_x_i[i.key] = m.addVar(lb=((-1) * x_i[i.key]) + 1, ub=num_levels -
                                    x_i[i.key], vtype=GRB.INTEGER,
                                    name='delta_x_i-%s' % (i.key))

    # x_i_pr (prime) - *new* level of switch i (from 1-4)
    x_i_pr = dict()
    for i in g['SWITCHES']:
        x_i_pr[i.key] = m.addVar(
            lb=min_sec_lvl, ub=max_sec_lvl, vtype=GRB.INTEGER, name='x_i_pr-%s' % (i.key))

    i_kl = extractLinkMatrix(g, nodesList)

    # Feasible links
    x_j_kl = dict()
    for j in flows:
        for k in [j.source] + g['SWITCHES']:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS'].keys():
                    x_j_kl[(j.key, k.key, l.key)] = m.addVar(
                        vtype=GRB.BINARY, name='x_j_kl-%s_%s_%s' % (j.key, k.key, l.key))

    xjkl_link_exists_indicator = dict()
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                xjkl_link_exists_indicator[(k.key, l.key)] = m.addVar(
                    vtype=GRB.BINARY, name='xjkl_link_exists_indicator-%s_%s' % (k.key, l.key))

    xjkl_level_indicator = dict()
    # Support variable to store the link level difference
    xjkl_level_difference = dict()
    # New (support) variable to enforce the (flow and min) level indicator
    abs_level_difference = dict()
    for j in flows:
        for k in nodesList:
            for l in nodesList:
                if (k.key, l.key) in g['LINKS']:
                    xjkl_level_indicator[(j.key, k.key, l.key)] = m.addVar(
                        vtype=GRB.BINARY, name='xjkl_level_indicator-%s-%s-%s' % (j.key, k.key, l.key))
                    xjkl_level_difference[(j.key, k.key, l.key)] = m.addVar(
                        lb=-(max_sec_lvl - min_sec_lvl), ub=(max_sec_lvl - min_sec_lvl), vtype=GRB.INTEGER,
                        name='xjkl_level_difference-%s-%s-%s' % (j.key, k.key, l.key))
                    abs_level_difference[(j.key, k.key, l.key)] = m.addVar(
                        lb=0, ub=(max_sec_lvl - min_sec_lvl),
                        vtype=GRB.INTEGER, name='abs_level_difference_indicator-%s-%s-%s' % (j.key, k.key, l.key))

    # Change 1 - Same level indicator to ensure that only paths where all switches are of the same level are taken
    xjkl_same_level_indicator = dict()
    xjkl_min_level = dict()
    # Support variable to store the link level difference
    kl_level_difference = dict()
    # New (support) variable to enforce the link level indicator
    abs_kl_level_difference = dict()
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                # Add the min level variable
                xjkl_min_level[(k.key, l.key)] = m.addVar(
                    vtype=GRB.INTEGER, name='xjkl_min_level-%s_%s' % (k.key, l.key))
                xjkl_same_level_indicator[(k.key, l.key)] = m.addVar(
                    vtype=GRB.BINARY, name='xjkl_same_level_indicator-%s_%s' % (k.key, l.key))
                kl_level_difference[(k.key, l.key)] = m.addVar(
                    lb=-(max_sec_lvl - min_sec_lvl), ub=(max_sec_lvl - min_sec_lvl),
                    vtype=GRB.INTEGER, name='kl_level_difference_indicator-%s_%s' % (k.key, l.key))
                abs_kl_level_difference[(k.key, l.key)] = m.addVar(
                    lb=0, ub=(max_sec_lvl - min_sec_lvl),
                    vtype=GRB.INTEGER, name='abs_kl_same_level_indicator-%s_%s' % (k.key, l.key))

    i_delta = dict()
    for i in g['SWITCHES']:
        i_delta[i.key] = m.addVar(
            vtype=GRB.BINARY, name='i_delta-%s' % (i.key))

    y = dict()
    for j in flows:
        for k in [j.source] + g['SWITCHES']:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS']:
                    y[(j.key, k.key, l.key)] = m.addVar(
                        vtype=GRB.BINARY, name='y-%s_%s_%s' % (j.key, k.key, l.key))

    alpha = dict()
    for j in flows:
        alpha[j.key] = m.addVar(vtype=GRB.BINARY, name='alpha_%s' % (j.key))

    m.update()

    ###### Step 3 - set objective ######
    m.setObjective(
        quicksum(j.demand * alpha[j.key] * flow_weight(j.level) for j in flows), GRB.MAXIMIZE)

    # Default time limit is infinity. Setting time limit to thirty minutes
    """ This is a problem only when M is really small but not (directly) infeasible or the solver doesnt think so at least.
          """
    m.Params.timeLimit = 1800

    ###### Step 4 - add constraints ######

    # General variable constraints
    for i in g['SWITCHES']:  # enforce the new level to be the starting level plus some delta
        m.addConstr(x_i_pr[i.key] == (
            x_i[i.key] + delta_x_i[i.key]), 'constr_x_i_pr-%s' % (i.key))

    # enforce new x_j_kl
    link_exists_indicators_added = set()
    # Change 2
    same_level_indicators_added = set()
    min_link_level_vars_added = set()
    fixed_host_levels = {h.key: h.level for h in g['HOSTS']}
    for j in flows:
        for k in nodesList:
            for l in nodesList:
                if (j.key, k.key, l.key) in x_j_kl.keys():
                    # diff flows can reuse this indicator
                    if (k.key, l.key) not in link_exists_indicators_added:
                        if (k.key, l.key) not in failed_links:
                            # force this indicator to be the correct value from link matrix
                            m.addConstr(xjkl_link_exists_indicator[(
                                k.key, l.key)] == int(i_kl[k.key][l.key]),
                                'constr_xjkl_link_exists_indicator-%s_%s' % (k.key, l.key))
                        else:
                            # force this indicator to be 0 as the link has failed
                            m.addConstr(xjkl_link_exists_indicator[(
                                k.key, l.key)] == int(0),
                                'constr_xjkl_link_exists_indicator-%s_%s' % (k.key, l.key))
                        link_exists_indicators_added.add((k.key, l.key))
                    lvl_check_dict_k = get_info_type(
                        k.key, x_i_pr, fixed_host_levels, fstr=False)
                    lvl_check_dict_l = get_info_type(
                        l.key, x_i_pr, fixed_host_levels, fstr=False)

                    # reuse for flows that may travel along this link
                    if (k.key, l.key) not in min_link_level_vars_added:
                        # Need to check if (SW,SW) or (Src,SW) or (SW,Dst) here (Maybe we can add this to the common function)
                        if k in g['SWITCHES'] and l in g['HOSTS']:
                            m.addConstr(xjkl_min_level[(k.key, l.key)] ==
                                        lvl_check_dict_k[k.key], name='constr_xjkl_min_level-%s_%s' % (k.key, l.key))
                        elif k in g['HOSTS'] and l in g['SWITCHES']:
                            m.addConstr(xjkl_min_level[(k.key, l.key)] ==
                                        lvl_check_dict_l[l.key], name='constr_xjkl_min_level-%s_%s' % (k.key, l.key))
                        else:
                            m.addConstr(xjkl_min_level[(k.key, l.key)] == min_(
                                lvl_check_dict_k[k.key], lvl_check_dict_l[l.key]),
                                name='constr_xjkl_min_level-%s_%s' % (k.key, l.key))
                        min_link_level_vars_added.add((k.key, l.key))

                    # Store the level difference (flow and min level)
                    m.addConstr(xjkl_level_difference[(j.key, k.key, l.key)] == (
                        xjkl_min_level[(k.key, l.key)] - j.level),
                        name='constr_xjkl_level_difference-%s_%s_%s' % (j.key, k.key, l.key))

                    # Store the absolute value of the level difference (flow and min level)
                    m.addConstr(abs_level_difference[(j.key, k.key, l.key)] == abs_(
                        xjkl_level_difference[(j.key, k.key, l.key)]),
                        name='constr_abs_jkl_level-%s_%s_%s' % (j.key, k.key, l.key))

                    # Forces xjkl_level_indicator to be zero when min level > flow level
                    m.addConstr(BIG_M * (1 - xjkl_level_indicator[(j.key, k.key, l.key)]) >= (
                        xjkl_min_level[(k.key, l.key)] - j.level))

                    # Forces  xjkl_level_indicator to be zero when flow level > min level
                    m.addConstr(-BIG_M * (1 - xjkl_level_indicator[(j.key, k.key, l.key)]) <= (
                        xjkl_min_level[(k.key, l.key)] - j.level))

                    # Use this to  prevent false positives (Can also use abs indicator here)
                    m.addConstr((xjkl_level_indicator[(j.key, k.key, l.key)] == 1) >> (
                        j.level == xjkl_min_level[(k.key, l.key)]),
                        'constr_xjkl_level_indicator-%s_%s_%s' % (j.key, k.key, l.key))

                    # Transposition - using this with big M will not affect correctness
                    m.addConstr((xjkl_level_indicator[(j.key, k.key, l.key)] == 0) >> (
                        abs_level_difference[(j.key, k.key, l.key)] >= 1),
                        'constr_xjkl_level_indicator-%s_%s_%s' % (j.key, k.key, l.key))

                    if k in g['SWITCHES'] and l in g['SWITCHES']:
                        # Change 3 - To ensure switches (links) are of the same level
                        if (k.key, l.key) not in same_level_indicators_added:
                            # Store the level difference (link)
                            m.addConstr(kl_level_difference[(k.key, l.key)] == (
                                lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key]),
                                name='constr_kl_level_difference-%s_%s' % (k.key, l.key))

                            # Store the absolute value of the link level difference
                            m.addConstr(abs_kl_level_difference[(k.key, l.key)] == abs_(
                                kl_level_difference[(k.key, l.key)]),
                                name='constr_abs_kl_level-%s_%s' % (k.key, l.key))

                            # Forces xjkl_same_level_indicator to be zero when k's level > l's level
                            m.addConstr((lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key]) <= (
                                BIG_M * (1 - xjkl_same_level_indicator[(k.key, l.key)])))

                            # Forces xjkl_same_level_indicator to be zero when l's level > k's level
                            m.addConstr((lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key]) >= (
                                -BIG_M * (1 - xjkl_same_level_indicator[(k.key, l.key)])))

                            # Use this to prevent false positives
                            m.addConstr((xjkl_same_level_indicator[(k.key, l.key)] == 1) >> (
                                lvl_check_dict_k[k.key] == lvl_check_dict_l[l.key]),
                                'constr_xjkl_same_level_indicator-%s_%s' % (k.key, l.key))

                            # Transposition - using this with big M will not affect correctness
                            m.addConstr((xjkl_same_level_indicator[(k.key, l.key)] == 0) >> (
                                abs_kl_level_difference[(k.key, l.key)] >= 1),
                                'constr_xjkl_same_level_indicator-%s_%s' % (k.key, l.key))
                            same_level_indicators_added.add((k.key, l.key))

                        m.addConstr(x_j_kl[(j.key, k.key, l.key)] == and_(xjkl_link_exists_indicator[(
                            k.key, l.key)], xjkl_level_indicator[(j.key, k.key, l.key)],
                            xjkl_same_level_indicator[(k.key, l.key)]),
                            'constr_xjkl-%s_%s_%s' % (j.key, k.key, l.key))

                    # Entry or exit link so they don't have to be of the same level i.e host and switch as the flow level
                    else:
                        m.addConstr(x_j_kl[(j.key, k.key, l.key)] == and_(xjkl_link_exists_indicator[(
                            k.key, l.key)], xjkl_level_indicator[(j.key, k.key, l.key)]),
                            'constr_xjkl-%s_%s_%s' % (j.key, k.key, l.key))

    # enforce i_delta
    abs_delta_x = dict()
    for i in g['SWITCHES']:
        abs_delta_x[i.key] = m.addVar(
            vtype=GRB.INTEGER, name='abs_delta_x-%s' % i.key)
        m.addConstr(abs_delta_x[i.key] == abs_(delta_x_i[i.key]))

        # Sets/forces i_delta[sw] to be 1 if delta_x_i[sw] != 0
        m.addConstr((BIG_M * i_delta[i.key]) >= abs_delta_x[i.key])

        # Sets/forces i_delta[sw] to be 0 if delta_x_i[sw] = 0
        m.addConstr((BIG_M * (1 - i_delta[i.key])) >= (1 - abs_delta_x[i.key]))

        m.addConstr((i_delta[i.key] == 0) >> (
            delta_x_i[i.key] == 0), 'constr_i_delta-%s' % (i.key))

    # Constraint 1
    for j in flows:
        for k in g['SWITCHES'] + [j.source]:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS']:
                    m.addConstr(y[(j.key, k.key, l.key)] <= x_j_kl[(
                        j.key, k.key, l.key)], 'constr1-%s_%s_%s' % (j.key, k.key, l.key))

    # Constraint 2 - (In flow)
    for j in flows:
        m.addConstr(
            quicksum(y[(j.key, j.source.key, k.key)] for k in g['SWITCHES'] if (
                j.key, j.source.key, k.key) in y.keys())
            == alpha[j.key], 'constr2-%s' % (j.key))

    # Constraint 3 - (Out flow)
    for j in flows:
        m.addConstr(
            quicksum(y[(j.key, k.key, j.dest.key)]
                     for k in g['SWITCHES'] if (j.key, k.key, j.dest.key) in y.keys())
            == alpha[j.key], 'constr3-%s' % (j.key))

    # Constraint 4 (Conservation)
    for j in flows:
        for l in g['SWITCHES']:
            m.addConstr(quicksum(y[(j.key, k.key, l.key)] for k in [j.source] + g['SWITCHES'] if
                                 (j.key, k.key, l.key) in y.keys()) == quicksum(
                y[(j.key, l.key, _m.key)] for _m in g['SWITCHES'] + [j.dest] if (j.key, l.key, _m.key) in y.keys()),
                'constr4-%s_%s' % (j.key, l.key))

    # Constraint 5 - No loops
    for j in flows:
        for l in g["SWITCHES"]:
            m.addConstr(quicksum(y[(j.key, k.key, l.key)] for k in [j.source] + g["SWITCHES"] if (
                j.key, k.key, l.key) in y.keys()) <= 1, 'constr5-%s' % (j.key))

    # Constraint 6 (Sum of demands for all flows entering switch i (from anywhere) is less than or equal to switch capacity)
    for i in g['SWITCHES']:
        m.addConstr(
            quicksum(quicksum(j.demand * y[(j.key, k.key, i.key)] for j in flows if (j.key, k.key, i.key) in y.keys())
                     for k in nodesList) <= i.capacity, 'constr6-%s' % (i.key))

    # Constraint 7 (Sum of flow demands using this link do not exceed link capacity)
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                m.addConstr(
                    quicksum(j.demand * y[(j.key, k.key, l.key)] for j in flows if (j.key, k.key, l.key) in y.keys()) <=
                    g['LINKS'][(k.key, l.key)].capacity,
                    'constr7-%s_%s' % (k.key, l.key))

    # Constraint 9 (limit the number of switches whose level changes)
    m.addConstr(quicksum(i_delta[i.key]
                         for i in g['SWITCHES']) <= M, 'constr9')

    # m.write("relmodel.lp")
    return m, y, alpha, x_i_pr, x_i, M, x_j_kl, xjkl_link_exists_indicator, xjkl_level_indicator, None, xjkl_min_level


def _solve_op_three(g, nodesList, flows, M, time_epoch):
    """ Optimization Problem 3 - Both routing up and routing down
         Objective changed (Minimize the number of transfer downs)
         X_j_k_l - No min constraint anymore
         Switch Capacity Constraint changed - as delta C_j - Penalty (Cost for routing down) - paid at the first hop and at the last hop
         No constraint 8
    """

    ## Preprocessing of sorts
    failed_links = [l[0] for l in failed_links]

    ###### Step 1 - create model ######
    m = Model('Relabeling')
    m.reset(0)

    ###### Step 2 - add variables ######
    x_i = dict()  # x_i - starting level of switch i
    for i in g['SWITCHES']:
        x_i[i.key] = i.level

    delta_x_i = dict()
    for i in g['SWITCHES']:
        if time_epoch < num_warmup_epochs:
            delta_x_i[i.key] = m.addVar(
                lb=0, ub=0, vtype=GRB.INTEGER, name='delta_x_i-%s' % (i.key))
        else:
            delta_x_i[i.key] = m.addVar(lb=((-1) * x_i[i.key]) + 1, ub=num_levels -
                                        x_i[i.key], vtype=GRB.INTEGER,
                                        name='delta_x_i-%s' % (i.key))

    # x_i_pr (prime) - *new* level of switch i (from 1-4)
    x_i_pr = dict()
    for i in g['SWITCHES']:
        x_i_pr[i.key] = m.addVar(
            lb=min_sec_lvl, ub=max_sec_lvl, vtype=GRB.INTEGER, name='x_i_pr-%s' % (i.key))

    i_kl = extractLinkMatrix(g, nodesList)

    # Feasible links
    x_j_kl = dict()
    for j in flows:
        for k in [j.source] + g['SWITCHES']:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS'].keys():
                    x_j_kl[(j.key, k.key, l.key)] = m.addVar(
                        vtype=GRB.BINARY, name='x_j_kl-%s_%s_%s' % (j.key, k.key, l.key))

    xjkl_link_exists_indicator = dict()
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                xjkl_link_exists_indicator[(k.key, l.key)] = m.addVar(
                    vtype=GRB.BINARY, name='xjkl_link_exists_indicator-%s_%s' % (k.key, l.key))

    xjkl_transfer_down_ok_indicator = dict()
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                xjkl_transfer_down_ok_indicator[(k.key, l.key)] = m.addVar(
                    vtype=GRB.BINARY, name='xjkl_transfer_down_ok_indicator-%s_%s' % (k.key, l.key))

    xjkl_transfer_down_indicator = dict()
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                xjkl_transfer_down_indicator[(k.key, l.key)] = m.addVar(
                    vtype=GRB.BINARY, name='xjkl_transfer_down_indicator-%s_%s' % (k.key, l.key))

    i_delta = dict()
    for i in g['SWITCHES']:
        i_delta[i.key] = m.addVar(
            vtype=GRB.BINARY, name='i_delta-%s' % (i.key))

    y = dict()
    for j in flows:
        for k in [j.source] + g['SWITCHES']:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS']:
                    y[(j.key, k.key, l.key)] = m.addVar(
                        vtype=GRB.BINARY, name='y-%s_%s_%s' % (j.key, k.key, l.key))

    alpha = dict()
    for j in flows:
        alpha[j.key] = m.addVar(vtype=GRB.BINARY, name='alpha_%s' % (j.key))

    m.update()

    ###### Step 3 - Set objective ######

    # Minimize the number of transfer downs
    m.setObjective(
        quicksum(quicksum(
            y[(j.key, k.key, l.key)] * xjkl_transfer_down_indicator[(k.key, l.key)] for k in [j.source] + g['SWITCHES']
            for l in g['SWITCHES'] + [j.dest] if (k.key, l.key) in g['LINKS']) for j in flows), GRB.MINIMIZE)

    # Default time limit is infinity. Setting time limit to thirty minutes.
    """ Gurobi tries to minimize the objective to zero.
          Increase the time limit when running a single specific case """
    m.Params.timeLimit = 1800

    ###### Step 4 -  Constraints ######

    # General variable constraints
    for i in g['SWITCHES']:  # enforce the new level to be the starting level plus some delta
        m.addConstr(x_i_pr[i.key] == (
            x_i[i.key] + delta_x_i[i.key]), 'constr_x_i_pr-%s' % (i.key))

    # enforce new x_j_kl
    link_exists_indicators_added = set()
    transfer_down_indicators_added = set()
    transfer_down_ok_indicators_added = set()
    fixed_host_levels = {h.key: h.level for h in g['HOSTS']}
    for j in flows:
        for k in nodesList:
            for l in nodesList:
                if (j.key, k.key, l.key) in x_j_kl.keys():

                    if (k.key, l.key) not in link_exists_indicators_added:
                        m.addConstr(xjkl_link_exists_indicator[(
                            k.key, l.key)] == int(i_kl[k.key][l.key]),
                            'constr_xjkl_link_exists_indicator-%s_%s' % (k.key, l.key))
                        link_exists_indicators_added.add((k.key, l.key))

                    lvl_check_dict_k = get_info_type(
                        k.key, x_i_pr, fixed_host_levels, fstr=False)
                    lvl_check_dict_l = get_info_type(
                        l.key, x_i_pr, fixed_host_levels, fstr=False)

                    if (k.key, l.key) not in transfer_down_ok_indicators_added:
                        # Sets/forces the transfer down  okay indicator to be 1
                        m.addConstr((BIG_M * xjkl_transfer_down_ok_indicator[(k.key, l.key)]) >= (
                            1 + delta_j - (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key])))

                        # Sets/forces the transfer down okay indicator to be 0
                        m.addConstr((BIG_M * (1 - xjkl_transfer_down_ok_indicator[(k.key, l.key)])) >= (
                            (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key]) - delta_j))

                        m.addConstr((xjkl_transfer_down_ok_indicator[(k.key, l.key)] == 1) >> (
                            lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key] <= delta_j),
                            'constr_xjkl_transfer_down_ok_indicator-%s_%s' % (k.key, l.key))
                        transfer_down_ok_indicators_added.add(
                            (k.key, l.key))

                    if (k.key, l.key) not in transfer_down_indicators_added:
                        # Sets/forces the transfer down indicator to be 1
                        m.addConstr((BIG_M * xjkl_transfer_down_indicator[(k.key, l.key)]) >= (
                            (lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key])),
                            'constr_xjkl_transfer_down_indicator-%s-%s' % (k.key, l.key))

                        m.addConstr((BIG_M * (1 - xjkl_transfer_down_indicator[(k.key, l.key)])) >= (
                            (1 + lvl_check_dict_l[l.key] - lvl_check_dict_k[k.key])),
                            'constr_xjkl_transfer_down_indicator-%s-%s' % (k.key, l.key))

                        m.addConstr((xjkl_transfer_down_indicator[(k.key, l.key)] == 0) >> (
                            lvl_check_dict_k[k.key] - lvl_check_dict_l[l.key] <= 0),
                            'constr_xjkl_transfer_down_indicator-%s_%s' % (k.key, l.key))
                        transfer_down_indicators_added.add((k.key, l.key))

                    m.addConstr(x_j_kl[(j.key, k.key, l.key)] == and_(xjkl_link_exists_indicator[(
                        k.key, l.key)], xjkl_transfer_down_ok_indicator[k.key, l.key]),
                        'constr_xjkl-%s_%s_%s' % (j.key, k.key, l.key))

    # enforce i_delta
    abs_delta_x = dict()
    for i in g['SWITCHES']:
        # abs_delta_x[i.key] = m.addVar(vtype=GRB.INTEGER, name='abs_delta_x-%s' % i.key)
        # m.addConstr(abs_delta_x[i.key] == abs_(delta_x_i[i.key]))
        #
        # # Sets/forces i_delta[sw] to be 1 if delta_x_i[sw] != 0
        # m.addConstr((BIG_M * i_delta[i.key]) >= abs_delta_x[i.key])
        #
        # # Sets/forces i_delta[sw] to be 0 if delta_x_i[sw] = 0
        # m.addConstr((BIG_M * (1 - i_delta[i.key])) >= (1 - abs_delta_x[i.key]))

        m.addConstr((i_delta[i.key] == 0) >> (
            delta_x_i[i.key] == 0), 'constr_i_delta-%s' % (i.key))

    # Constraint 1
    for j in flows:
        for k in g['SWITCHES'] + [j.source]:
            for l in g['SWITCHES'] + [j.dest]:
                if (k.key, l.key) in g['LINKS']:
                    m.addConstr(y[(j.key, k.key, l.key)] <= x_j_kl[(
                        j.key, k.key, l.key)], 'constr1-%s_%s_%s' % (j.key, k.key, l.key))

    # Constraint 2 - (In flow)
    for j in flows:
        m.addConstr(
            quicksum(y[(j.key, j.source.key, k.key)] for k in nodesList if (
                j.key, j.source.key, k.key) in y.keys())
            == 1, 'constr2-%s' % (j.key))

    # Constraint 3 - (Out flow)
    for j in flows:
        m.addConstr(
            quicksum(y[(j.key, k.key, j.dest.key)]
                     for k in nodesList if (j.key, k.key, j.dest.key) in y.keys())
            == 1, 'constr3-%s' % (j.key))

        # To set alpha
        m.addConstr(alpha[j.key] == quicksum(y[(j.key, k.key, j.dest.key)] for k in g['SWITCHES'] if (
            j.key, k.key, j.dest.key) in y.keys()), 'meta-constr1-%s' % (j.key))

    # Constraint 4 (Conservation)
    for j in flows:
        for l in nodesList:
            m.addConstr(quicksum(y[(j.key, k.key, l.key)] for k in nodesList if
                                 (j.key, k.key, l.key) in y.keys()) == quicksum(
                y[(j.key, l.key, _m.key)] for _m in nodesList + [j.dest] if (j.key, l.key, _m.key) in y.keys()),
                'constr4-%s_%s' % (j.key, l.key))

    # Constraint 5 - No loops
    for j in flows:
        for l in g['SWITCHES']:
            m.addConstr(quicksum(y[(j.key, k.key, l.key)] for k in nodesList if (
                j.key, k.key, l.key) in y.keys()) <= 1, 'constr5-%s' % (j.key))

    # Constraint 6 (Sum of demands for all flows entering switch i (from anywhere) is less than or equal to switch capacity)
    for i in g['SWITCHES']:
        m.addConstr(
            quicksum(quicksum(j.demand * y[(j.key, k.key, i.key)] for j in flows if (j.key, k.key, i.key) in y.keys())
                     for k in nodesList)
            +
            quicksum(quicksum(delta_c_j * y[(j.key, i.key, j.dest.key)] * y[(j.key, k.key, i.key)] for j in flows if (
                j.key, k.key, i.key) in y.keys() and (j.key, i.key, j.dest.key) in y.keys()) for k in
                nodesList)
            +
            quicksum(delta_c_j * y[(j.key, j.source.key, i.key)] for j in flows if
                     (j.key, j.source.key, i.key) in y.keys()) <= i.capacity, 'constr6-%s' % (i.key))

    # Constraint 7 (Sum of flow demands using this link do not exceed link capacity)
    for k in nodesList:
        for l in nodesList:
            if (k.key, l.key) in g['LINKS']:
                m.addConstr(
                    quicksum(j.demand * y[(j.key, k.key, l.key)] for j in flows if (j.key, k.key, l.key) in y.keys()) <=
                    g['LINKS'][(k.key, l.key)].capacity,
                    'constr7-%s_%s' % (k.key, l.key))

    # Constraint 9 (limit the number of switches whose level changes)
    m.addConstr(quicksum(i_delta[i.key]
                         for i in g['SWITCHES']) <= M, 'constr9')

    # m.write("relmodel.lp")
    return m, y, alpha, x_i_pr, x_i, M, x_j_kl


def run_solver(g, nodesList, flows, network_type, m_perc, time_epoch, switches_pos_dict, hosts_pos_dict, failed_links,
               switches_whos_level_changed_last_run):
    if time_epoch <= relabeling_period:
        recent_relabelling_epoch = relabeling_period
    else:
        recent_relabelling_epoch = int(
            time_epoch / relabeling_period) * relabeling_period

    print("Current epoch : ", time_epoch,
          " Last relabelling epoch: ", recent_relabelling_epoch)
    switches_whos_level_changed = set()
    # set switch lvl change limit first
    if time_epoch < num_warmup_epochs or time_epoch != recent_relabelling_epoch:
        # Note (QB): RG changes: reuse old set and dont let anymore switches change until it is another relabeling epoch
        M = 0
        switches_whos_level_changed = switches_whos_level_changed_last_run
    else:
        # otherwise keep the empty set and let switches change new levels (and init reboot time)
        M = int(len(g['SWITCHES']) * m_perc)



    if (optimization_problem == 1):
        m, y, alpha, x_i_pr, x_i, M, x_j_kl, xjkl_link_exists_indicator, xjkl_level_indicator, xjkl_transfer_down_ok_indicator, xjkl_min_level = _solve_op_one(
            g, nodesList, flows, M, time_epoch, failed_links)
    elif (optimization_problem == 2):
        m, y, alpha, x_i_pr, x_i, M, x_j_kl, xjkl_link_exists_indicator, xjkl_level_indicator, xjkl_transfer_down_ok_indicator, xjkl_min_level = _solve_op_two(
            g, nodesList, flows, M, time_epoch, failed_links)
    # elif (optimization_problem == 3):
    #     m, y, alpha, x_i_pr, x_i, M, x_j_kl = _solve_op_three(
    #         g, nodesList, flows, M, time_epoch)
    else:
        print("bad opt number")
        exit(1)

    ###### Step 5 - solve ######
    start_time = time.time_ns()
    m.optimize()
    finish_time = time.time_ns()

    ###### Step 6 - print results ######
    flows_dict = {f.key: f for f in flows}

    # In order to plot number of flows routed vs number of flows for each level (Using dict in the rare case that we have
    # non contiguous levels
    flowLevelCoverage = {level: 0 for level in range(
        min_sec_lvl, max_sec_lvl + 1)}

    if m.status == GRB.Status.OPTIMAL:

        # get updated link capacities (grab from constraint 7)
        print("\n===== Link capacity results:")
        new_link_caps = dict()

        for k in nodesList:
            for l in nodesList:
                if (k.key, l.key) in g['LINKS']:
                    used_cap = 0
                    for flow_key in flows_dict.keys():
                        if (flow_key, k.key, l.key) in y.keys():
                            if alpha[flow_key].x == 1:  # if this flow is routed
                                # if this link is used for this flow
                                if y[(flow_key, k.key, l.key)].x == 1:
                                    used_cap += flows_dict[flow_key].demand
                    new_link_caps[(k.key, l.key)] = g['LINKS'][(
                        k.key, l.key)].capacity - used_cap
                    print("[(%s, %s)]: old cap (%.2f Gb) -> residual cap (%.2f Gb)" % (k.key, l.key,
                                                                                       g['LINKS'][(
                                                                                           k.key,
                                                                                           l.key)].capacity / 1e9,
                                                                                       new_link_caps[
                                                                                           (k.key, l.key)] / 1e9))

        # get updated switch capacities (grab from constraint 6)
        print("\n===== Switch capacity results:")
        new_sw_caps = dict()
        for sw in g['SWITCHES']:
            used_cap = 0
            # for k in g["SOURCES"] + g["SWITCHES"]:
            for k in nodesList:
                for flow_key in flows_dict.keys():
                    if (flow_key, k.key, sw.key) in y.keys():
                        if alpha[flow_key].x == 1:  # if this flow is routed
                            # if this link (into switch) is used for this flow
                            # If this switch is not rebooting then count used capacity
                            if y[(flow_key, k.key, sw.key)].x == 1 and sw.wait_time_remaining==0:
                                used_cap += flows_dict[flow_key].demand
            new_sw_caps[sw.key] = g['SWITCHES'][switches_pos_dict[sw.key]
                                                ].capacity - used_cap
            print("[%s]: old cap (%.2f Gb) -> residual cap (%.2f Gb)" % (sw.key,
                                                                         g['SWITCHES'][
                                                                             switches_pos_dict[sw.key]].capacity / 1e9,
                                                                         new_sw_caps[sw.key] / 1e9))

        # get updated switch levels
        new_switch_levels = dict()

        print("\n===== Switch level results:")
        num_switches_with_level_change = 0
        level_change_delta = 0
        for key in x_i_pr.keys():
            print("old level of [%s]: %d -> new level of [%s]: %d" %
                  (key, x_i[key], key, x_i_pr[key].x))
            if x_i[key] != round(x_i_pr[key].x):
                new_switch_levels[key] = x_i_pr[key].x
                num_switches_with_level_change += 1
                switches_whos_level_changed.add(key)
                level_change_delta += x_i[key] - x_i_pr[key].x
                # Set waiting time (Only when relabelled)
                g['SWITCHES'][switches_pos_dict[key]
                              ].wait_time_remaining = reboot_time



        print("\n===== Flow coverage results:")
        no_flows_covered = 0
        path_length = dict()
        transfer_downs = {fk: 0 for fk in flows_dict.keys()}
        flow_paths = {flow_key: list() for flow_key in flows_dict.keys()}
        fixed_host_levels = {h.key: h.level for h in g['HOSTS']}
        blocked_flows = None
        flows_routed_after = set()
        waiting_flows = set()
        for flow_key in flows_dict.keys():
            is_a_waiting_flow = False
            if round(alpha[flow_key].x) == 1.0:  # if flow was routed

                ##################################################
                # trace backwards instead
                curr_node = flows_dict[flow_key].dest.key
                flow_paths[flow_key].insert(0, curr_node)
                seen = set()
                while 1:
                    for y_key in y.keys():
                        # for each link for this flow
                        # if (y_key[0] == flow_key):
                        #     print(y_key," : ", round(y[y_key].x))
                        if (y_key[0] == flow_key) and (round(y[y_key].x) == 1.0) and (y_key[2] == curr_node) and (
                                y_key[1] not in seen):  # NOTE (QB; 7/27/2020): tracing path backwards
                            # we have bidirectional links so this just found the link going in the opposite direction and curr_node got stuck and didnt find anything else, using seen to prevent this (might still give bad paths tho)
                            print("link found (%s, %s)" % (y_key[1], y_key[2]))
                            # If the switch is rebooting
                            if y_key[1] in switches_pos_dict.keys():
                                print(
                                    y_key[1], " - ", g['SWITCHES'][switches_pos_dict[y_key[1]]].wait_time_remaining)
                            if y_key[1] in switches_pos_dict.keys() and g['SWITCHES'][
                                    switches_pos_dict[y_key[1]]].wait_time_remaining > 0:
                                is_a_waiting_flow = True
                                waiting_flows.add(flows_dict[flow_key])
                                flows_dict[flow_key].wait_time_remaining = max(
                                    flows_dict[flow_key].wait_time_remaining,
                                    g['SWITCHES'][switches_pos_dict[y_key[1]]].wait_time_remaining)
                                if blocked_flows is None:  # dont count in coverage and add it so it gets retried next epoch
                                    blocked_flows = set()
                                blocked_flows.add(flows_dict[flow_key])
                            #     break
                            # else:
                            flow_paths[flow_key].insert(0, y_key[1])
                            # this will become the y_key[1] in the next iteration
                            seen.add(y_key[2])
                            curr_node = y_key[1]
                            if curr_node == flows_dict[flow_key].source.key:
                                break
                    if curr_node == flows_dict[flow_key].source.key or is_a_waiting_flow:
                        break

                if is_a_waiting_flow:
                    print("flow[%s] (level: %s)  from %s (level: %s) -> %s (level: %s) is waiting" %
                          (flow_key, flows_dict[flow_key].level, flows_dict[flow_key].source.key,
                           flows_dict[flow_key].source.level, flows_dict[flow_key].dest.key,
                           flows_dict[flow_key].dest.level))

                else:
                    path_length[flow_key] = len(flow_paths[flow_key])
                    curr_node = flows_dict[flow_key].source.key
                    path_str = curr_node + \
                        "(lev: %d)" % (
                            flows_dict[flow_key].source.level) + " -> "
                    print("flow[%s] (level: %s) was routed from %s (level: %s) -> %s (level: %s)" %
                          (flow_key, flows_dict[flow_key].level, flows_dict[flow_key].source.key,
                           flows_dict[flow_key].source.level, flows_dict[flow_key].dest.key,
                           flows_dict[flow_key].dest.level))
                    for i in range(1, len(flow_paths[flow_key])):
                        if (optimization_problem == 2):
                            if flow_paths[flow_key][i] == flows_dict[flow_key].dest.key:
                                # dont count as tx down for opt2
                                path_str += flow_paths[flow_key][i] + \
                                    "(lev: %d)" % (
                                    flows_dict[flow_key].dest.level)
                            elif curr_node == flows_dict[flow_key].source.key:
                                # dont count as tx down for opt2
                                path_str += flow_paths[flow_key][i] + "(lev:%d)" % (
                                    x_i_pr[flow_paths[flow_key][i]].x) + " -> "
                            else:
                                path_str += flow_paths[flow_key][i] + "(lev:%d)" % (
                                    x_i_pr[flow_paths[flow_key][i]].x) + " -> "
                                lvl_check_dict_next_curr_node = get_info_type(
                                    curr_node, x_i_pr, fixed_host_levels, fstr=False)
                                lvl_check_dict_i = get_info_type(
                                    flow_paths[flow_key][i], x_i_pr, fixed_host_levels, fstr=False)
                                if (lvl_check_dict_next_curr_node is fixed_host_levels) and (
                                        lvl_check_dict_i is x_i_pr):
                                    if fixed_host_levels[curr_node] > round(x_i_pr[flow_paths[flow_key][i]].x):
                                        transfer_downs[flow_key] += 1
                                elif (lvl_check_dict_next_curr_node is x_i_pr) and (lvl_check_dict_i is x_i_pr):
                                    if round(x_i_pr[curr_node].x) > round(x_i_pr[flow_paths[flow_key][i]].x):
                                        transfer_downs[flow_key] += 1
                                elif (lvl_check_dict_next_curr_node is x_i_pr) and (
                                        lvl_check_dict_i is fixed_host_levels):
                                    if round(x_i_pr[curr_node].x) > fixed_host_levels[flow_paths[flow_key][i]]:
                                        transfer_downs[flow_key] += 1
                        else:  # for opt1 can have tx down at entry or exit link
                            if flow_paths[flow_key][i] == flows_dict[flow_key].dest.key:
                                # dont count as tx down for opt2
                                path_str += flow_paths[flow_key][i] + \
                                    "(lev: %d)" % (
                                    flows_dict[flow_key].dest.level)
                            else:
                                path_str += flow_paths[flow_key][i] + "(lev:%d)" % (
                                    x_i_pr[flow_paths[flow_key][i]].x) + " -> "
                                lvl_check_dict_next_curr_node = get_info_type(
                                    curr_node, x_i_pr, fixed_host_levels, fstr=False)
                                lvl_check_dict_i = get_info_type(
                                    flow_paths[flow_key][i], x_i_pr, fixed_host_levels, fstr=False)
                                if (lvl_check_dict_next_curr_node is fixed_host_levels) and (
                                        lvl_check_dict_i is x_i_pr):
                                    if fixed_host_levels[curr_node] > round(x_i_pr[flow_paths[flow_key][i]].x):
                                        transfer_downs[flow_key] += 1
                                elif (lvl_check_dict_next_curr_node is x_i_pr) and (lvl_check_dict_i is x_i_pr):
                                    if round(x_i_pr[curr_node].x) > round(x_i_pr[flow_paths[flow_key][i]].x):
                                        transfer_downs[flow_key] += 1
                                elif (lvl_check_dict_next_curr_node is x_i_pr) and (
                                        lvl_check_dict_i is fixed_host_levels):
                                    if round(x_i_pr[curr_node].x) > fixed_host_levels[flow_paths[flow_key][i]]:
                                        transfer_downs[flow_key] += 1
                        curr_node = flow_paths[flow_key][i]
                    print("Path: %s" % path_str)
                    print("Number of transfer downs: %d\n===" %
                          transfer_downs[flow_key])

                    flows_routed_after.add(flows_dict[flow_key])
                    no_flows_covered += 1
                    flowLevelCoverage[flows_dict[flow_key]
                                      .level] = flowLevelCoverage[flows_dict[flow_key].level] + 1

                    if (optimization_problem == 2) and (transfer_downs[flow_key] > 0):
                        print("1 or more tx downs for opt2")
                        exit(1)

                ##################################################
            else:  # flow wasnt routed
                if blocked_flows is None:
                    blocked_flows = set()
                blocked_flows.add(flows_dict[flow_key])

        # Count number of routing downs
        total_num_route_downs_after = 0
        for j in flows:
            if round(alpha[j.key].x) == 1.0:  # if the flow was routed
                total_num_route_downs_after += get_num_route_downs(
                    g, j, flow_paths[j.key])
        if no_flows_covered > 0:
            avg_num_route_downs_after = total_num_route_downs_after * 1.0 / no_flows_covered
        else:
            avg_num_route_downs_after = 0.0

        print("\n===== Switch activity:")
        seen = list()
        for sw_key in x_i_pr.keys():
            num_flows_traversing = 0
            for flow_key in flows_dict.keys():
                if round(alpha[flow_key].x) == 1.0:  # if the flow was routed
                    # if the flow crosses this switch at all
                    if sw_key in flow_paths[flow_key]:
                        num_flows_traversing += 1
            print("[%s] has [%s] flows traversing it (level change: %s)" % (sw_key + ' ' * (5 - len(sw_key)), str(
                num_flows_traversing) + ' ' * (2 - len(str(num_flows_traversing))),
                sw_key in switches_whos_level_changed))

        # # num_unique_disrupted_flows = 0 # if a flow touches a switch whos level changed, it was disrupted; edit "waiting" not disrupted
        # for flow_key in flows_dict.keys():
        #     if round(alpha[flow_key].x) == 1.0:  # if the flow was routed
        #         for sw_key in x_i_pr.keys():
        #             if sw_key in switches_whos_level_changed:
        #                 # does flow touch this switch whos level changed?
        #                 if sw_key in flow_paths[flow_key]:
        #                     if flow_key not in waiting_flows:  # only count the flow once
        #                         waiting_flows.add(flow_key)
        #                     # num_unique_disrupted_flows += 1

        print("\n===== Other metrics:")
        print("Optimization problem: %d" % optimization_problem)

        print("Average path length: %.2f hops" %
              np.mean([path_length[f] for f in path_length.keys()]))

        print("Number of switches with level change: %d (M=%d)" %
              (num_switches_with_level_change, M))

        print("Number of unique flows that are waiting (%d/%d): %.2f%%" %
              (len(waiting_flows), len(flows), len(waiting_flows) * 1.0 / len(flows) * 100))

        avg_lvl_change = level_change_delta * 1.0 / \
            (num_switches_with_level_change or 1)
        sign = "+" if avg_lvl_change > 0 else ""
        print("Average level change: %s%.2f" % (sign, avg_lvl_change))

        total_num_tx_downs_after = np.sum(
            [transfer_downs[flow_key] for flow_key in transfer_downs.keys() if round(alpha[flow_key].x) == 1.0])
        avg_num_tx_downs_after = np.mean(
            [transfer_downs[f] for f in transfer_downs.keys() if round(alpha[f].x) == 1.0])
        print("Average number of transfer downs: %.2f (B=%d)" %
              (avg_num_tx_downs_after, B))
        # print(transfer_downs)

        num_feasible_links = 0
        links_found = set()
        for j in flows:
            for k in nodesList:
                for l in nodesList:
                    if (j.key, k.key, l.key) in y.keys():
                        if x_j_kl[(j.key, k.key, l.key)].x == 1:
                            # print("x_j_kl[(%s, %s, %s)].x: %s" % (j.key, k.key, l.key, x_j_kl[(j.key, k.key, l.key)].x))
                            if (k.key, l.key) not in links_found:
                                num_feasible_links += 1
                                links_found.add((k.key, l.key))
        if len(g['LINKS']) > 0:
            feasible_links_percentage = (
                num_feasible_links * 1.0 / len(g['LINKS']) * 100)
            print("Feasible links percentage of total links: %.2f%%" %
                  feasible_links_percentage)

        flows_covered_percentage = (no_flows_covered * 1.0 / len(flows) * 100)
        print("Flows covered total (%d/%d): %.2f%%" %
              (no_flows_covered, len(flows), flows_covered_percentage))
        # print("flows_covered_by_level: ", flows_covered_by_level)
        # print('flows_total_by_level: ', flows_total_by_level)
        # print('flows_covered_by_level: ', flows_covered_by_level)
        # for lvl in range(1, max_sec_lvl+1):
        #     div = 0
        #     if flows_total_by_level[lvl] > 0:
        #         div = flows_covered_by_level[lvl] * \
        #             1.0/flows_total_by_level[lvl]*100
        #     print("Flows covered [level %d] (%d/%d): %.2f%%" %
        #           (lvl, flows_covered_by_level[lvl], flows_total_by_level[lvl], div))

        # Printing debug information
        if DEBUG and blocked_flows is not None and False:
            print("################## DEBUG INFORMATION ##################")
            i_kl = extractLinkMatrix(g, nodesList)
            feasiblePaths = [[] for f in blocked_flows]
            for flow in blocked_flows:
                # Print stats
                flow_key = flow.key
                print("\nFor flow: ", flow_key, "level: ", flow.level, "src: ", flow.source.key, flow.source.level,
                      "dest: ",
                      flow.dest.key, flow.dest.level)
                findFeasiblePaths(i_kl, flow, blocked_flows,
                                  nodesList, feasiblePaths)

                for path in feasiblePaths[blocked_flows.index(flow)]:
                    isPathFeasible = True
                    path_string = " "
                    number_of_transfer_downs = 0
                    if optimization_problem == 1:
                        local_j = delta_j
                        local_B = B
                        prev_level = flow.source.level
                    elif optimization_problem == 2:
                        local_j = 0
                        local_B = 0
                        prev_level = flow.level
                    for node in path[:-1]:
                        if node.key in x_i_pr.keys() and x_i_pr[node.key].X >= flow.level:
                            # print(node.key, type(node.key))
                            current_level = int(round(x_i_pr[node.key].X))
                            if prev_level - current_level > local_j:
                                # print("\tThis path violated delta j")
                                isPathFeasible = False
                                break
                            elif current_level < prev_level:
                                number_of_transfer_downs = number_of_transfer_downs + 1

                            if number_of_transfer_downs > local_B:
                                # print("\tThis path violated B")
                                isPathFeasible = False
                                break

                            if current_level != prev_level and optimization_problem == 2:
                                isPathFeasible = False
                                break

                            path_string = path_string + node.key + \
                                "(" + str(x_i_pr[node.key].X) + ")" + "->"
                            prev_level = current_level

                        elif node is not flow.source:
                            if node == flow.dest:
                                print(node.key)
                            else:
                                # Some other flow's source or dest
                                if node.level < flow.level or prev_level - node.level > local_j:
                                    isPathFeasible = False
                                    break

                    if isPathFeasible:
                        path_string = flow.source.key + "->" + path_string
                        path_string += flow.dest.key
                        print("Path(feasible): ", path_string)

                print("\n\n")
                for link in g["LINKS"]:
                    if (flow_key, link[0], link[1]) in x_j_kl.keys() and x_j_kl[(flow_key, link[0], link[1])].X > 0.0:
                        print("X_", flow_key, "_", link[0], "_", link[1], ": ", x_j_kl[(
                            flow_key, link[0], link[1])].X)
                    else:
                        print(link)
                        print(" Link exists: ",
                              xjkl_link_exists_indicator[(link[0], link[1])].X)
                        if (flow_key, link[0], link[1]) in xjkl_level_indicator:
                            print(" Level indicator: ", xjkl_level_indicator[(
                                flow_key, link[0], link[1])].X)
                            if xjkl_level_indicator[(flow_key, link[0], link[1])].X == 0:
                                print("Flow level: ", flow.level)
                                if link[0] in x_i_pr.keys():
                                    print(" ", link[0], x_i_pr[link[0]].X)
                                else:
                                    print(" ", link[0], x_i[link[0]])

                                if link[1] in x_i_pr.keys():
                                    print(" ", link[1], x_i_pr[link[1]].X)
                                else:
                                    print(" ", link[1], x_i[link[1]])

                        print(" Min level indicator:  ",
                              xjkl_min_level[(link[0], link[1])].X)
                        if optimization_problem == 1:
                            print(" Transfer down ok indicator:",
                                  xjkl_transfer_down_ok_indicator[(link[0], link[1])].X)
                            if xjkl_transfer_down_ok_indicator[(link[0], link[1])].X == 0:
                                if link[0] in x_i_pr.keys():
                                    print(" ", link[0], x_i_pr[link[0]].X)
                                else:
                                    print(" ", link[0], x_i[link[0]])

                                if link[1] in x_i_pr.keys():
                                    print(" ", link[1], x_i_pr[link[1]].X)
                                else:
                                    print(" ", link[1], x_i[link[1]])

                print("\n\n")

        exec_time = ((finish_time - start_time) * 1.0 / 1e9)
        print("\033[93m{}\033[00m".format(
            "[Execution time for opt: %.3fs]" % exec_time))

        print("[Feasible model]")

        # # IMPORTANT (5/26): Update levels and capacities now
        # print("Updating topology capacities and levels")
        if new_switch_levels is not None:
            for key in new_switch_levels.keys():
                g['SWITCHES'][switches_pos_dict[key]].level = int(
                    round(new_switch_levels[key]))
        # if new_sw_caps is not None:
        #     for key in new_sw_caps.keys():
        #         g['SWITCHES'][switches_pos_dict[key]].capacity = new_sw_caps[key]

        # print("alphas: ", [
        #     alpha[flow_key].x for flow_key in flows_dict.keys()])

        # print("xjkl_link_exists_indicator: ", [(k.key, l.key, xjkl_link_exists_indicator[(
        #     k.key, l.key)].x) for k in nodesList for l in nodesList if (k.key, l.key) in g['LINKS'].keys()])

        # print("xjkl_level_indicator: ",
        #       [(j.key, k.key, l.key, xjkl_level_indicator[(j.key, k.key, l.key)].x) for j in flows for k in [
        #           j.source] + g['SWITCHES'] for l in g['SWITCHES'] + [j.dest] if (k.key, l.key) in g['LINKS'].keys()])

        if blocked_flows is not None:
            print("blocked_flows: ", ["(%s, %d, %s->%s), " % (str(f.key),
                                                              f.level, f.source.key, f.dest.key) for f in
                                      blocked_flows])

        print("failed_links: ", failed_links)
        # for key in run_info_end['topo_changes']['new_link_caps'].keys(): # changed to dont update here and dont reclaim in randomly_depart_flows
        #     run_info_end['g']['LINKS'][key].capacity = run_info_end['topo_changes']['new_link_caps'][key]

        # if blocked_flows is not None:
        #     print("blocked_flows: ", [str(f.key) + "," + str(f.level) +
        #                               "(" + f.source.key + "->" + f.dest.key + ")" for f in blocked_flows])
        # topo_changes = {'new_switch_levels': new_switch_levels,
        #                 'new_sw_caps': new_sw_caps, 'new_link_caps': new_link_caps,
        #                 'flow_level_coverage': flowLevelCoverage, 'flow_paths': flow_paths,
        #                 'transfer_downs': np.mean([transfer_downs[f] for f in transfer_downs.keys() if alpha[f].x == 1]), "objective_value": objective_value}
        # return flows_covered_percentage, exec_time, topo_changes, blocked_flows

        # empty conflict doesnt affect departing flows but cant be None
        # conflict_switches_for_this_flow_empty = {flow_key: set() for flow_key in flows_dict.keys()}
        # flows_failing_tx_down_delta_after = list()
        # flows_failing_tx_down_num_after = list()

        # flows_routed_phase1, num_flows_routed_phase1, num_flows_sorted_phase1, flows_routed_phase3, num_flows_routed_phase3, num_flows_routed_phase3_by_level, num_flows_sorted_phase3, running_time, flow_paths_phase3, conflict_switches_for_this_flow_phase3, blocked_flows_phase1, blocked_flows_phase3, flows_failing_tx_down_delta_phase1, flows_failing_tx_down_delta_phase3, flows_failing_tx_down_num_phase1, flows_failing_tx_down_num_phase3, waiting_flows_phase3, \
        # avg_num_tx_downs_phase3[num_flows_idx][samp_idx][m_perc_index][time_epoch_idx], \
        # total_num_tx_downs_phase3[num_flows_idx][samp_idx][m_perc_index][time_epoch_idx], \
        # avg_num_route_downs_phase3[num_flows_idx][samp_idx][m_perc_index][time_epoch_idx], \
        # total_num_route_downs_phase3[num_flows_idx][samp_idx][m_perc_index][
        #     time_epoch_idx], num_switches_with_level_change_phase3, switches_whos_level_changed_phase3, rc
        return list(), 0, len(flows), flows_routed_after, no_flows_covered, flowLevelCoverage, len(
            flows), exec_time, flow_paths, None, None, blocked_flows, None, None, None, None, waiting_flows, avg_num_tx_downs_after, total_num_tx_downs_after, avg_num_route_downs_after, total_num_route_downs_after, None, switches_whos_level_changed, 0
    else:
        print("model not optimal")

        exec_time = ((finish_time - start_time) * 1.0 / 1e9)
        print("\033[93m{}\033[00m".format(
            "[Execution time for opt: %.3fs]" % exec_time))

        # exit(1)
        # flowLevelCoverage_empty = {level: 0 for level in range(min_sec_lvl, max_sec_lvl + 1)}
        # flow_paths_empty = {flow_key: list() for flow_key in flows_dict.keys()}
        # conflict_switches_for_this_flow_empty = {flow_key: set() for flow_key in flows_dict.keys()}
        return list(), 0, len(flows), list(), 0, None, len(
            flows), exec_time, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, -1

    """
    elif (m.status == GRB.Status.TIME_LIMIT):
        print("===Time Limit ==")
        objective_value = m.objVal

        # get updated link capacities (grab from constraint 7)
        print("\n===== Link capacity results:")
        new_link_caps = dict()

        for k in nodesList:
            for l in nodesList:
                if (k.key, l.key) in g['LINKS']:
                    used_cap = 0
                    for flow_key in flows_dict.keys():
                        if (flow_key, k.key, l.key) in y.keys():
                            if alpha[flow_key].x == 1:  # if this flow is routed
                                    # if this link is used for this flow
                                if y[(flow_key, k.key, l.key)].x == 1:
                                    used_cap += flows_dict[flow_key].demand
                    new_link_caps[(k.key, l.key)] = g['LINKS'][(
                        k.key, l.key)].capacity - used_cap
                    print("[(%s, %s)]: old cap (%.2f Gb) -> residual cap (%.2f Gb)" % (k.key, l.key,
                                                                                       g['LINKS'][(
                                                                                           k.key,
                                                                                           l.key)].capacity / 1e9,
                                                                                       new_link_caps[
                                                                                           (k.key, l.key)] / 1e9))

        # get updated switch capacities (grab from constraint 6)
        print("\n===== Switch capacity results:")
        switches_dict = {g['SWITCHES'][i].key: i for i in range(
            len(g['SWITCHES']))}  # map key to list pos
        new_sw_caps = dict()
        for i in g['SWITCHES']:
            used_cap = 0
            for k in g["SOURCES"] + g["SWITCHES"]:
                for flow_key in flows_dict.keys():
                    if (flow_key, k.key, i.key) in y.keys():
                        if alpha[flow_key].x == 1:  # if this flow is routed
                            # if this link (into switch) is used for this flow
                            if y[(flow_key, k.key, i.key)].x == 1:
                                used_cap += flows_dict[flow_key].demand
            new_sw_caps[i.key] = g['SWITCHES'][switches_dict[i.key]
                                               ].capacity - used_cap
            print("[%s]: old cap (%.2f Gb) -> residual cap (%.2f Gb)" % (i.key,
                                                                         g['SWITCHES'][
                                                                             switches_dict[i.key]].capacity / 1e9,
                                                                         new_sw_caps[i.key] / 1e9))

        # get updated switch levels
        new_switch_levels = dict()

        print("\n===== Switch level results:")
        num_switches_with_level_change = 0
        switches_whos_level_changed = list()
        level_change_delta = 0
        for key in x_i_pr.keys():
            print("old level of [%s]: %d -> new level of [%s]: %d" %
                  (key, x_i[key], key, x_i_pr[key].x))
            if x_i[key] != x_i_pr[key].x:
                new_switch_levels[key] = x_i_pr[key].x
                num_switches_with_level_change += 1
                switches_whos_level_changed.append(key)
                level_change_delta += x_i[key] - x_i_pr[key].x

        print("\n===== Flow coverage results:")
        flows_covered = 0
        path_length = dict()
        transfer_downs = {fk: 0 for fk in flows_dict.keys()}
        flow_paths = {flow_key: list() for flow_key in flows_dict.keys()}
        blocked_flows = None
        for flow_key in flows_dict.keys():
            plen = 0
            # flows_total_by_level[flows_dict[flow_key].level] += 1
            if round(alpha[flow_key].x) == 1.0:  # if flow was routed
                flows_covered += 1
                # Flow Level Coverage: Flow ID to (Flow level, Route Status)
                flowLevelCoverage[flows_dict[flow_key]
                                  .level] = flowLevelCoverage[flows_dict[flow_key].level] + 1
                print("flow[%s] (level: %s) was routed from %s (level: %s) -> %s (level: %s)" %
                      (flow_key, flows_dict[flow_key].level, flows_dict[flow_key].source.key,
                       flows_dict[flow_key].source.level, flows_dict[flow_key].dest.key,
                       flows_dict[flow_key].dest.level))
                next_nd = dict()
                for y_key in y.keys():
                    if y_key[0] == flow_key:  # for each link for this flow
                        if y[y_key].x != 1 and round(y[y_key].x) == 1.0:
                            print("GUROBI NON BINARY", y_key,
                                  y[y_key].x, round(y[y_key].x))
                        if round(y[y_key].x) == 1.0:  # if the link is used
                            next_nd[y_key[1]] = y_key[2]

                nd = flows_dict[flow_key].source.key
                path_str = nd + " -> "
                flow_paths[flow_key].append(nd)
                nd = next_nd[nd]
                seen = set()  # TODO: this shouldn't break anything? need this to stop infinite loop sometimes
                while nd not in seen:
                    plen += 1
                    if nd == flows_dict[flow_key].dest.key:  # next_nd hop is dest
                        path_str += nd
                        flow_paths[flow_key].append(nd)
                        break

                    else:  # current node is a switch and next_nd hop is a switch
                        if (nd in next_nd) and (next_nd[nd] in [sw.key for sw in g['SWITCHES']]):
                            if x_i_pr[nd].x > x_i_pr[next_nd[nd]].x:
                                # is a transfer down to next_nd hop switch
                                transfer_downs[flow_key] += 1
                        elif (nd in next_nd) and (next_nd[nd] in [ds for ds in g['DESTS']]):
                            if x_i_pr[nd].x > x_i[next_nd[nd]]:
                                # is a transfer down to next_nd hop switch
                                transfer_downs[flow_key] += 1
                        path_str += nd + (" (lev %s->%d)" %
                                          (x_i[nd], x_i_pr[nd].x)) + " -> "
                        flow_paths[flow_key].append(nd)
                        seen.add(nd)

                        if (nd in next_nd):
                            nd = next_nd[nd]
                            if nd in seen:
                                print(alpha[flow_key].x)
                                print("Loop/Seen: ", nd, " ", seen)
                        else:
                            print(path_str, "\n ND:\t ", nd)
                            print(next_nd.keys())
                            exit("no next_nd")

                print("Path: %s" % path_str)
                print("Number of transfer downs: %d\n===" %
                      transfer_downs[flow_key])

                path_length[flow_key] = plen
            else:  # flow wasnt routed
                if blocked_flows is None:
                    blocked_flows = list()
                blocked_flows.append(flows_dict[flow_key])

        print("\n===== Switch activity:")
        seen = list()
        for sw_key in x_i_pr.keys():
            num_flows_traversing = 0
            for flow_key in flows_dict.keys():
                if alpha[flow_key].x == 1:  # if the flow was routed
                    # if the flow crosses this switch at all
                    if sw_key in flow_paths[flow_key]:
                        num_flows_traversing += 1
            print("[%s] has [%s] flows traversing it (level change: %s)" % (sw_key + ' ' * (5 - len(sw_key)), str(
                num_flows_traversing) + ' ' * (2 - len(str(num_flows_traversing))),
                sw_key in switches_whos_level_changed))

        waiting_flows = list()
        # num_unique_disrupted_flows = 0 # if a flow touches a switch whos level changed, it was disrupted; edit "waiting" not disrupted
        for flow_key in flows_dict.keys():
            if alpha[flow_key].x == 1:  # if the flow was routed
                for sw_key in x_i_pr.keys():
                    if sw_key in switches_whos_level_changed:
                        # does flow touch this switch whos level changed?
                        if sw_key in flow_paths[flow_key]:
                            if flow_key not in waiting_flows:  # only count the flow once
                                waiting_flows.add(flow_key)

        print("\n===== Other metrics:")
        print("Optimization problem: %d" % optimization_problem)

        print("Average path length: %.2f hops" %
              np.mean([path_length[f] for f in path_length.keys()]))

        print("Number of switches with level change: %d (M=%d)" %
              (num_switches_with_level_change, M))

        print("Number of unique flows that are waiting (%d/%d): %.2f%%" %
              (len(waiting_flows), len(flows), len(waiting_flows) * 1.0 / len(flows) * 100))

        avg_lvl_change = level_change_delta * 1.0 / \
            (num_switches_with_level_change or 1)
        sign = "+" if avg_lvl_change > 0 else ""
        print("Average level change: %s%.2f" % (sign, avg_lvl_change))

        print("Average number of transfer downs: %.2f (B=%d)" %
              (np.mean([transfer_downs[f] for f in transfer_downs.keys() if alpha[f].x == 1]), B))
        # print(transfer_downs)

        num_feasible_links = 0
        links_found = set()
        for j in flows:
            for k in nodesList:
                for l in nodesList:
                    if (k.key, l.key) in g['LINKS']:
                        if k.key != j.dest.key and l.key != j.source.key:
                            if x_j_kl[(j.key, k.key, l.key)].x == 1:
                                # print("x_j_kl[(%s, %s, %s)].x: %s" % (j.key, k.key, l.key, x_j_kl[(j.key, k.key, l.key)].x))
                                if (k.key, l.key) not in links_found:
                                    num_feasible_links += 1
                                    links_found.add((k.key, l.key))
        if len(g['LINKS']) > 0:
            feasible_links_percentage = (
                num_feasible_links * 1.0 / len(g['LINKS']) * 100)
            print("Feasible links percentage of total links: %.2f%%" %
                  feasible_links_percentage)

        flows_covered_percentage = (flows_covered * 1.0 / len(flows) * 100)
        print("Flows covered total (%d/%d): %.2f%%" %
              (flows_covered, len(flows), flows_covered_percentage))

        exec_time = ((finish_time - start_time) * 1.0 / 1e9)
        print("\033[93m{}\033[00m".format(
            "[Execution time for opt: %.3fs]" % exec_time))

        print("[Feasible model]")
        if blocked_flows is not None:
            print("blocked_flows: ", [str(f.key) + "," + str(f.level) +
                                      "(" + f.source.key + "->" + f.dest.key + ")" for f in blocked_flows])

        topo_changes = {'new_switch_levels': new_switch_levels,
                        'new_sw_caps': new_sw_caps, 'new_link_caps': new_link_caps,
                        'flow_level_coverage': flowLevelCoverage, 'flow_paths': flow_paths,
                        'transfer_downs': np.mean([transfer_downs[f] for f in transfer_downs.keys() if alpha[f].x == 1]), "objective_value": objective_value}

        return flows_covered_percentage, m.Params.timeLimit, topo_changes, blocked_flows
    elif (m.status == GRB.Status.SUBOPTIMAL):
        objective_value = 0
        flows_covered = 0
        flows_covered_percentage = 0
        blocked_flows = list()
        for flow_key in flows_dict.keys():
            try:
                if round(alpha[flow_key].x) == 1.0:  # if flow was routed
                    flows_covered += 1
                    flowLevelCoverage[flows_dict[flow_key]
                                      .level] = flowLevelCoverage[flows_dict[flow_key].level] + 1
                else:
                    blocked_flows.append(flows_dict[flow_key])
            except Exception:
                flows_covered += 0

        flows_covered_percentage = (flows_covered * 1.0 / len(flows) * 100)
        print("Flows covered:", flows_covered_percentage)

        topo_changes = {'new_switch_levels': None,
                        'new_sw_caps': None, "flow_paths": None, 'new_link_caps': None,
                        'flow_level_coverage': flowLevelCoverage, "objective_value": objective_value}

        print("===SUB OPTIMAL===")
        exec_time = ((finish_time - start_time) * 1.0 / 1e9)
        print("\033[93m{}\033[00m".format(
            "[Execution time for opt: %.3fs]" % exec_time))
        return flows_covered_percentage, exec_time, topo_changes, blocked_flows
    else:
        objective_value = 0
        flows_covered = 0
        flows_covered_percentage = 0
        blocked_flows = list()
        for flow_key in flows_dict.keys():
            try:
                if round(alpha[flow_key].x) == 1.0:  # if flow was routed
                    flows_covered += 1
                    flowLevelCoverage[flows_dict[flow_key]
                                      .level] = flowLevelCoverage[flows_dict[flow_key].level] + 1
                else:
                    blocked_flows.append(flows_dict[flow_key])
            except Exception:
                flows_covered += 0

        flows_covered_percentage = (flows_covered * 1.0 / len(flows) * 100)
        print("Flows covered:", flows_covered_percentage)

        topo_changes = {'new_switch_levels': None,
                        'new_sw_caps': None, "flow_paths": None, 'new_link_caps': None,
                        'flow_level_coverage': flowLevelCoverage, "objective_value": objective_value}

        print("=== INFEASIBLE MODEL===", m.status)
        return 0, m.Params.timeLimit, topo_changes, blocked_flows
    """
