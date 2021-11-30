from NodeAndEdge import Node,Edge
import xml.etree.ElementTree as ET
import os
import numpy as np
import pandas as pd
import pickle
from Graph import Graph, creategraph


if __name__=='__main__':
    filelist = [file for file in os.listdir('Instances/')]
    #graphs = [creategraph('Instances/' + file) for file in os.listdir('Instances/')]
    #graph = creategraph('Instances/' + filelist[2])
    #print(graph)

    num_teams = []
    timeslot = []
    solutionsize = []
    num_node = []
    num_constraints_edges = []
    num_hard = []
    num_soft = []
    num_forced = []
    names = []
    for i in range(len(filelist)):
    #for i in range(3):
        graph = creategraph('Instances/' + filelist[i])

        num_teams.append(len(graph.teams))
        timeslot.append(len(graph.slots))
        solutionsize.append(graph.solutionsize)

        num_node.append(len(graph.nodedict))
        hard_edge = 0
        soft_edge = 0
        for node in graph.nodedict:
            #print(node)
            hard_edge += len(graph.nodedict[node].edges_hard) + len(graph.nodedict[node].edges_hard_complex)
            soft_edge += len(graph.nodedict[node].edges_soft) + len(graph.nodedict[node].edges_soft_complex)

        soft_tot = np.round(soft_edge / 2)
        hard_tot = np.round(hard_edge / 2)

        num_constraints_edges.append(soft_tot + hard_tot)
        num_hard.append(hard_tot)
        num_soft.append(soft_tot)
        num_forced.append(len(graph.forcedselections))

        names.append(filelist[i])

        print(i)
    df = pd.DataFrame({'Instance Name': names,
                       'Number of Teams': num_teams,
                       'Number of Slots': timeslot,
                       'Solution Size': solutionsize,
                       'Nodes': num_node,
                       'Total Constraint Edges': num_constraints_edges,
                       'Hard Constraint Edges': num_hard,
                       'Soft Constraint Edges': num_soft,
                       'Forced Selections': num_forced})

    df.to_csv('GraphSummary.csv')
    #df = pd.DataFrame({'Instance Name': names})
    #df.to_csv('Names.csv')


