#!/usr/bin/env python3.8

"""
    Main entry point. Performs the following functions
    1. Starts mini net network with specified topology
    2. Can generate flows as required 
    Topologies containing loops and controllers which support these loops
   
"""

from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.node import RemoteController, DefaultController, OVSKernelSwitch, OVSSwitch
from functools import partial
import time

from topo import FatTreeTopo, MeshTopo, StarWANTopo, SimpleTopo
from flow import run_experiment



def setup_network_and_start_experiment():
    "Create and start a network using a specific topology"
    topo = FatTreeTopo(6)
    # net = Mininet(topo, switch=partial(OVSKernelSwitch,protocols="OpenFlow14"),controller=partial(RemoteController, ip='127.0.0.1', port=6653))
    net = Mininet(topo, switch=partial(OVSSwitch,protocols="OpenFlow10"),  controller=partial(
        RemoteController, ip='127.0.0.1', port=6633))
    # net=Mininet(topo)
    net.start()
    time.sleep(30)
    # print("Done")
    # print("Dumping switch connections")
    # dumpNodeConnections(net.switches)
    # print("Dumping host connections")
    # dumpNodeConnections(net.hosts)
    print("Testing network connectivity")
    net.pingAll()
    run_experiment(net,topo)
    net.stop()

if __name__ == '__main__':
    # Tell mininet to print useful information
    setLogLevel('info')
    setup_network_and_start_experiment()
