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

# Flow generation related constants 
NUM_FLOWS=48
# How often do new flows arrive (Also epoch duration?)
FLOW_GENERATION_PERIOD=60
# Total experimentation time = (# Epochs* Flow generation period)
TOTAL_EXPERIMENTATION_EPOCHS=10
MIN_FLOW_DURATION=110

burst_flag = False  # simulate bursts
burst_start_idx = 1220
burst_duration = 700 # Indefinite
burst_period = 1220
# not 0 indexed; set high to ignore; should be >warmup
burst_ratio = 0.9  # ratio of flows arriving that have burst_lvl
burst_lvl = 4  # level of burst flows


# simulate link fails
LINK_FAILURE_FLAG = False  
link_fails_start_idx = 1220
# Number of "epochs"  must be; < link_fail_period
LINK_FAILURE_DURATION = 1  
# Every third "epoch"
LINK_FAILURE_PERIOD = 4
LINK_FAILURE_PERC = 0.10

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

min_flow_demand = 50e1  # bps
max_flow_demand = 50e1
agg_to_core_link_cap = 100
edge_to_agg_link_cap = 100
host_to_edge_link_cap = 100
mesh_switch_link_cap = 100
default_link_cap = 100
switch_cap = 100



