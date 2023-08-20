#!/usr/bin/env python3.7
#
# rm_solve.py
#
# Description   : Attempting to use the RM algorithm
# Created by    :
# Date          : December 2019
# Last Modified : December 2019


""" 
    There are nine constraints
    Network Constraints:
	    1. Can only route flows through feasible links
	    2. In flow
	    3. Out flow
	    4. Conservation
	    5. No loops
	    6. Sum of demands for all flows entering switch i (from anywhere) is less than or equal to switch capacity)
	    7. Sum of flow demands using this link do not exceed link capacity
    MLS Constraints: (Constant (Atoms) constraints)
	    8. Number of transfer downs 
	    9. NUmber of switches whose level changes

	    """


from common import extractLinkMatrix, B, M, num_levels, min_sec_lvl, max_sec_lvl, delta_j
from common import genRandomTopo


def extractFeasibleLinksMatrix(g, lm_tmp, flow):
    """Extract feasible links from link matrix (for heuristic algorithm)."""

    compactNodesList = g["SWITCHES"] + [flow.source] + [flow.dest]
    flm = {r.key: {c.key: str(0) for c in compactNodesList}
           for r in compactNodesList}

    # fill matrix
    for r_node in compactNodesList:
        for c_node in compactNodesList:
            if (lm_tmp[r_node.key][c_node.key] == str(1)) and (flow.level <= min(r_node.level, c_node.level) and (r_node.level-c_node.level <= delta_j)):  # new condition
                flm[r_node.key][c_node.key] = str(1)
            else:
                flm[r_node.key][c_node.key] = str(0)

    return flm


def computePaths(flm, tmp, flow, flows, nodesList, feasiblePaths, path):
    # DP - DFS
    row = flm[tmp.key]
    nextNode = None
    path.append(tmp)

    # Check if we've reached the dest
    if tmp.key == flow.dest.key:
        feasiblePaths[flows.index(flow)].append(path)
        return

    # Check the row to see the reachable vertices
    for j in row:

        # Find the node
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
    computePaths(flm, flow.source, flow, flows, nodesList, feasiblePaths, [])
    # Worst case feasible paths is empty
    printFeasiblePaths(feasiblePaths, flow, flows)
    return


def printFeasiblePaths(feasiblePaths, flow, flows):
    for path in feasiblePaths[flows.index(flow)]:
        print("Path: ", end=' ')
        for node in path[:-1]:
            print(node.key, end='->')
        print(flow.dest.key)


def isItAFeasiblePath(path, flowLevel):
    for i in range(1, len(path)-1):
        print(path[i])
        # new condition
        if (not(flowLevel <= min(path[i].level, path[i+1].level) and (path[i].level - path[i+1].level <= delta_j))):
            return False
    return True


def enforceNetworkConstraints():
    pass


def enforceMLSConstraints():
    pass


def run_rm_solver(g, nodesList, flows):
    feasiblePaths = [[] for f in flows]
    allPaths = [[] for f in flows]
    switches = g["SWITCHES"]
    x_i = [switch.level for switch in switches]
    alpha_j = [0 for flow in flows]
    x_j_k_l = []
    y_j_k_l = [0]
    equations = []
    linkMatrix = extractLinkMatrix(g, nodesList)

    print(len(flows))

    # Some way to decide which flows should I check first?
    # For now let's just try to route flows with the highest demand*flow level
    sortedFlows = sorted(flows, key=lambda x: x.level, reverse=True)
    print(len(sortedFlows))
    pos = 0
    # Step 1 - Find all paths from flows and remove infeasible flows
    for flow in sortedFlows:
        pos = sortedFlows.index(flow)
        print(pos, " ", flow.source, " ", flow.dest, " ", flow.level)
        #feasibleLinkMatrix=extractFeasibleLinksMatrix(g, linkMatrix,flow)
        findFeasiblePaths(linkMatrix, flow, sortedFlows, nodesList, allPaths)
        # if len(allPaths[pos])==0:
        # 	print("No path found for a given flow. Source:",flow.source.key," Dest:",flow.dest.key)
        # 	sortedFlows.remove(flow)
        # 	allPaths.pop(pos)
        # elif isItAFeasiblePath(allPaths[pos],flow.level):
        #  print("Feasible path found for flow -  Source:",flow.source.key," Dest:",flow.dest.key)
        #  alpha_j[pos]=1
        #  sortedFlows.remove(flow)
        #  allPaths.pop(pos)
        #  continue

    # Step 2 - Pick a path for each flow (Need a better heuristic here)
    # Constaints of the type t<= Beta - two ways flow.level <= X_k and similarly flow.level <= X_l
    # X_k<=X_l+delta_j
    # The question is which path's constraints should be added (at least in the first flow)
    # Is there one such heuristic which will guarantee that we haven't missed a possible solution
    # Once we add the "perfect" path's constraints then we update all the other flow's paths constraints and then check if any of them are now feasible
    # Need to enforce the MLS constraints (B per flow and M)
    set(allPaths)
    for flow in sortedFlows:
        for paths in allPaths[sortedFlows.index(flow)]:
            maxLen = 1
            pos = -1
            for path in paths:
                if len(set(paths).intersection(allPaths)) > maxLen:
                    pos = paths.index(path)
                    maxLen = len(set(paths).intersection(allPathSet))
            print(paths[pos])
    pass


if __name__ == '__main__':

    num_flows1 = 10
    min_num_switches = 3
    max_num_switches = 7
    num_sources = 10
    # num_dests = 10

    g, nodesList, flows = genRandomTopo(
        max_num_switches,  num_sources, num_dests, num_flows1)
    run_rm_solver(g, nodesList, flows)
