### Globals ###
DEBUG = False
measure_agility = False
show_sec_metric_flag = False
optimization_problem = 1
delta_j = 2
delta_c_j = 50e6
# TODO - If we want a similar approach like the simulated run - Need a better approach
M_PERC=0.10
B = 5
min_sec_lvl = 1
max_sec_lvl = 4
num_levels = max_sec_lvl - min_sec_lvl + 1

# Relabeling related constants and variables
RELABELING_PERIOD = 100
# Warm up <= Relabeling
WARMUP_PERIOD = 180
# Reboot time
REBOOT_PERIOD=10


# balance_flows_flag = False  # set True and hosts_level_method=5/6 to balance # OPT1
# eq_src_dst_flag = True  # auto checks if doing OPT2
switches_level_method = 2  # 1-const, 2-random, 3-other
hosts_level_method = 3  # 1-const, 2-random, 3-other, 5-==sw_lvl, 6-subnet
flows_level_method = 5  # 1-const, 2-random, 5-==min(src,dst)
const_switch_lvl = max_sec_lvl
const_host_lvl = const_switch_lvl
const_flow_lvl = const_switch_lvl
hosts_per_wan_switch = 5
hosts_per_mesh_switch = 5

weight_of_rebooting_switch = 10e6
flow_weight_power = 2


# KEYS USED IN GRAPH DICT
SWITCHES = "SWITCHES"
LINKS = "LINKS"
HOSTS = "HOSTS"
NODES_LIST = "nodesList"
# Added because controller needs this (Until we come up with something better)
SWITCH_MAC_TO_DPID="SWITCH_MAC_TO_DPID"


# Classes/DS necessary for heuristic API (Will be maintained by Discovery)
class MLSSwitch():
    """A switch node in the network."""

    def __init__(self, key, level, port_info=None,capacity=float("inf")):
        # Should be DPID
        self.key = key
        self.level = level
        #  Dict of port no to MAC
        self.port_info=port_info
        # These two will probably not be used
        self.capacity = capacity
        self.wait_time_remaining = -1


class MLSLink():
    """A link between sources, switches, or destinations in the network.
        Source - Object (Switch/host). Dest same
        Added port info might be useful for rule installation
        No port info for hosts
    """

    def __init__(self, source_obj, dest_obj, source_port=None,dest_port=None,capacity=float("inf")):
        self.k = source_obj
        self.l = dest_obj
        self.source_port=source_port
        self.dest_port=dest_port
        # Will not be used (enforced by Mininet already)
        self.capacity = capacity


class MLSHost():
    """A host node in the network."""

    def __init__(self, key, level):
        # MAC address
        self.key = key
        self.level = level


class Flow():
    """A flow of packets originating from a specified source toward a specified destination, with given level and demand (in bps).
        Source/Dest - ID because there are two components 

    """

    def __init__(self, key, source, dest, level):
        self.key = key
        # DPID or MAC
        self.source = source
        # DPID or MAC
        self.dest = dest
        self.level = level
        # Controller will not need these details
        # self.demand = demand
        # self.retries = 0
        # self.duration = duration
        # self.wait_time_remaining = -1


# Utility functions needed by heuristic

def get_info_type(node, switches, hosts, return_ds=False):
    """Returns the correct dictionary (or just the key) for indexing purposes in the solver/heuristic."""
    if node in switches:  # overload use for dicts and lists
        if return_ds:
            return switches
        else:
            return SWITCHES
    elif node in hosts:
        if return_ds:
            return hosts
        else:
            return HOSTS
    else:
        print("node (%s) is bad in get_info_type" % node)
        print("switches: ", switches)
        print("hosts: ", hosts)
        exit(1)
    return


def flow_weight(level):
    """Computes the weight of a flow given the level."""
    if level <= max_sec_lvl:
        return level ** flow_weight_power
    else:
        print("Bad level in flow_weight(): %d." % level)
        exit(1)