"""
    Utilities for post processing to plot graphs.
    1. Coverage graph
    2. Agility graph
    3. Flow latency graph
"""

from posixpath import split
import readline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = "output"
NETWORK_TYPE = "fat-tree"
OPTIMIZATION_PROBLEM = 1

def plot_graph(graph_type):
    fig = plt.figure()  # put all on same plot
    plt.rcParams["font.family"] = "Times New Roman"
    coverage_data = read_coverage_data()
    print(coverage_data)
    line_sty = '-'
    line_color = 'red'
    m_perc = 0.10
    plt.plot([i for i in range(len(coverage_data))],
             coverage_data, linewidth=0.5, color=line_color, linestyle=line_sty,
             label='%s, M=%.2f' % ("Heuristic", m_perc))

    plt.xticks([i for i in range(len(coverage_data)) if i % 100 == 0], [
        str(i) for i in range(len(coverage_data)) if i % 100 == 0], fontsize=20)
    plt.yticks(fontsize=26)
    # plt.tick_params(direction='in')
    plt.xlabel('Time (s)', fontsize=26)
    plt.ylabel("Coverage (%)", fontsize=26)
    plt.ylim(0, 1)
    plt.grid(True, color='grey', linestyle=':',
             alpha=0.3, linewidth=0.25)
    # plt.title("Coverage over time", fontsize=12)
    plt.axvspan(240, 420, color='red', alpha=0.2)

    plt.legend(fontsize=20, loc='best')
    plt.tight_layout()
    plt.savefig('%s/fig-coverage-over-time-topo=%s-type=%s-opt=%d-num_flows=%s-num_sw_val=%s-M=%.2f.pdf' % (OUTPUT_DIR,
                                                                                                    NETWORK_TYPE,
                                                                                                    str(
                                                                                                        "Heuristic"),
                                                                                                    OPTIMIZATION_PROBLEM,
                                                                                                    str(
                                                                                                        8),
                                                                                                    str(
                                                                                                        6),
                                                                                                    m_perc))

    plt.show()


def read_coverage_data(file_name="coverage.txt"):
    coverage_data = []
    with open("coverage.txt", "r") as coverage_results:
        data=coverage_results.readline()

        while data:
            data=float(data.rstrip())
            coverage_data.append(data)
            data=coverage_results.readline()

    return coverage_data

def read_flow_data(file_name = "flow_out_new.txt"):

    with open("flow_out_new.txt","r") as flow_results:
        data = flow_results.readline()
        flow_data = [[],[],[],[],[],[],[],[],[],[]]
        index = 0
        time = 0
        flag = 0

        

        while data:
            if (flag == 1):
                break
            if (data.split(' ')[0] == 'EPOCH'):
                data  = flow_results.readline()
                # print(time)
                # time += 1
                # print(data)
                while (data.split(' ')[0] != 'EPOCH'):
                    try:
                        #print(data.split(' ')[-2].split("=")[-1])
                        latency = float(data.split(' ')[-2].split("=")[-1])
                        flow_data[index].append(latency/1000)
                        data = flow_results.readline()
                        # print(time)
                        # time += 1
                        #print("Good: " + str(flow_data))
                    except:
                        # print(time)
                        # time += 1
                        data = flow_results.readline()
                    if (data == ""):
                        flag = 1
                        break
                index = index + 1
            else:
                data = flow_results.readline()
                # print(time)
                # time += 1

        return flow_data

def get_packet_loss(file_name = "flow_out_new.txt"):

    with open("flow_out_new.txt","r") as flow_results:
        data = flow_results.readline()
        flow_data = [[],[],[],[],[],[],[],[],[],[]]
        index = 0
        time = 0
        flag = 0

        

        while data:
            if (flag == 1):
                break
            if (data.split(' ')[0] == 'EPOCH'):
                data  = flow_results.readline()
                
                #print(data)
                while (data.split(' ')[0] != 'EPOCH'):
                    if (data.split(' ')[0] == '---'):
                        data = flow_results.readline()
                        split_data = data.split(' ')
                        for i in range(len(split_data)):
                            if (split_data[i] == 'packet'):
                                break
                        try:
                            loss = float(data.split(' ')[i-1].strip('%'))
                            print(data.split(' ')[i-1])
                            flow_data[index].append(loss/100)
                            data = flow_results.readline()
                        except:
                            data = flow_results.readline()
                    elif (data == ""):
                        flag = 1
                        break
                    else:
                        data = flow_results.readline()
                index = index + 1
            else:
                data = flow_results.readline()
                print(time)
                time += 1
        
        return flow_data

def plot_flow(flow_data, packet_loss):
    # construct some data like what you have:
    fig = plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    epochs = []
    for epoch in range(1, len(flow_data) + 1):
        epochs.append(epoch)
    
    ep_ct = 1
    means = []

    #print(len(flow_data))
    for latency in flow_data:
        if(len(latency) != 0):
            latency = np.array(latency)
            
            min = np.min(latency)
            max = np.max(latency)
            mean = np.mean(latency)
            
            std = np.std(latency)
            means.append(mean)
            #pl.errorbar(ep_ct, mean, std, fmt='ok', lw=3)
            #plt.errorbar(ep_ct, mean, [[mean - min], [std]],
            # fmt='ok', ecolor='gray', lw=2)
            ep_ct += 1
        else:
            means.append(0)
            ep_ct += 1
            
    
    mean_packet_loss = []
    for loss in packet_loss:
        loss = np.array(loss)
        mean = np.mean(loss)
        
        mean_packet_loss.append(mean)

    df = pd.DataFrame({"Packet Loss": mean_packet_loss})
    df2 = pd.DataFrame({'data':means})
    
    ax = df.plot(kind="bar", alpha = 0.5, rot = 0)
    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'], fontsize = 20)
    ax2 = ax.twinx()
    ax.tick_params(axis='both', which='major', labelsize=15)
    #ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    #ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax2.plot(ax.get_xticks(),df2['data'].values,linestyle='-',marker='o', linewidth=2.0, color = 'k')
    ax2.tick_params(axis='both', which='major', labelsize=15)
    
    #ax.tick_params(axis='both', which='minor', labelsize=8)
    print("MEAN RTT:",means,np.mean(means))
    print("MEAN PACKET LOSS:",mean_packet_loss,np.mean(mean_packet_loss))
    #plt.plot(epochs, mean_packet_loss, kind = 'bar')
    #plt.bar(epochs, mean_packet_loss, color = 'b', alpha = 0.5)
    #plt.plot(epochs, means, '.r-') 
    ax.legend(loc = 'upper left')
    ax.set_ylabel('Packet Loss (%)', fontsize = 26)
    ax.set_xlabel('Epoch',  fontsize = 26)
    ax2.set_ylabel('Round-Trip Time (RTT)',  fontsize = 26)
    plt.tight_layout()
    plt.savefig('flow.pdf')

    

    '''
    epoch = [1,2,3,4,5]
    
    for elem in flow_data:
        elem = tuple(elem)
    
    for x,y in zip(epoch,flow_data):
        print(y)
        plt.scatter([x] * len(y), y)

    plt.xticks([1, 2, 3, 4, 5])
    plt.savefig('t.png')
    '''


# plot_graph(None)
loss = get_packet_loss()
flow = read_flow_data()
plot_flow(flow, loss)

