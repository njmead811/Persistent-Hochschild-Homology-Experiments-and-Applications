
"""
This file is an auxiliary file that contains functions computing persistent (Hochschild) homology. The main functions compute the following:

1. the homology of the directed flag complex of a single graph (Standard)
2. the homologies of the directed flag complexes of an appropriate filtration of graphs by weights (PersistentStandard)
3. the Hochschild homology (in the sense of Caputi-Riihimaki) of a single graph (Hochschild)
4. the Hochschild homology of the directed flag complexes of the reachability graphs in an edge weight filtration of a digraph (PersistentHochschild)
                                  

There are also several auxiliary functions related to calculating the filtration, as well as threshholding a graph. 

"""

# import necessary packages
import itertools
import numpy as np
import networkx as nx
import statistics 
import pandas as pd
import math
import pyflagser
import matplotlib.pyplot as plt

"""
 This function deletes the edges of weighted digraph of weight <= thresh and >= thresh2
 In some applications it is useful to delete the edges below a certain weight. For instance edges below a certain weight 
  might represent negative correlations which are irrelevant. 
"""
def Threshhold(G, thresh, thresh_2):
    # convert the graph to a numpy array and set the entries <= thresh and >= thresh2 to 0
    as_np = nx.to_numpy_array(G)
    as_np[as_np < thresh] = 0
    as_np[as_np > thresh_2] = 0
    # convert the numpy array back to a networkx digraph
    return nx.from_numpy_array(as_np, parallel_edges=False, create_using=nx.DiGraph)


                                   
"""
This function returns a list of weights for a digraph G from least to greatest. 
"""
def get_weights(G):
    # the set of edges of the graph G
    edge_set_G = G.edges()
    # the array that will contain the set of weights of the graph G
    weight_set = []
    # for each edge add its weight to weight_set
    for e in edge_set_G:
        weight_set.append(G.get_edge_data(e[0], e[1])['weight'])
    # there may be multiple edges with the same weight so get the set of unique weights    
    weight_set = list(set(weight_set))
    # sort the weights from least to greatest
    weight_set.sort()
   
    return weight_set

"""
Given a digraph G, this function returns an array consisting of the subgraphs of G spanned by edges of weight less than w for 
each w in the array weights 
"""  
def returnFiltration(G, weights):
    # the set of edges and nodes of G
    edge_set_G = G.edges()
    node_set_G = G.nodes()
    
    # the array representing the filtration, initialized to consist of empty graphs 
    filtration = [nx.DiGraph() for i in range(len(weights))]
    max_weight = max(weights)
    # add each node of the original graph to each graph in the filtration. The nodes remain the same for the original graph
    # and each element of the filtration 
    for i in range(len(weights)):
        for n in node_set_G:
            filtration[i].add_node(n)
    # compute the edges of the graphs in the filtration          
    for e in edge_set_G:
        # the weight of e in the original graph 
        curr_weight = G.get_edge_data(e[0], e[1])['weight']
        # add the edge to the appropriate elements of the filtration  
        for i in range(len(weights)):
            if curr_weight <= weights[i]:
                filtration[i].add_edge(e[0], e[1])
    return filtration


"""                                                      
A function that computes Hochschild Betti numbers of a graph G in dimensions 0, 1, 2 as well as the Euler characteristic
 e(G),  where e(G) H_{0}^{hoch}(G) - H_{1}^{hoch}(G) + H_{2}(G), where H_{i}^{hoch} is the ith Hochschild homology.  
"""
def Hochschild(G):
    # Hochschild homology is obtained by taking the ordinary homology of the directed flag complex óf the reachability poset
    condensed_graph = nx.transitive_closure(nx.condensation(G))
    # compute the homology of condensed_graph by first converting the weighted graph to a numpy array and then applying pyflagser. 
    matrix = nx.to_numpy_array(condensed_graph)
    homology = pyflagser.flagser_unweighted(matrix, min_dimension=0, max_dimension=2, directed = True, coeff = 2)
    # homology includes two attributes homology["betti"] an array of betti numbers and homology["euler"], which the invariant e(G) defined above
 
    # The array to be returned by the function. We set it to be the form [H_{0}^{hoch}(Fl(G)), H_{1}^{hoch}(Fl(G)), H_{2}^{hoch}(Fl(G)), e(G) ]
    betti_array = [0, 0, 0, 0]
    # we would expect that betti_array = homology[betti] + [homology["euler"]] would be sufficient to compute betti_array, but
    # we have to use the more convoluted code below to compute betti_array because if the directed flag complex is 1-dimensional, homology["betti"] returns a 2-dimensional array 
    betti_array[0:len(homology["betti"])] = homology["betti"]
    betti_array[3] = homology["euler"]
    
    return betti_array
    
"""
For a digraph G and a collection of weights, the graph will compute the filtration given by the collection of weights
 and return a 4 * len(weights) array containing the Hochschild Homology and Betti numbers of each of the elements of the filtration. 
"""
def PersistentHochschild(G, weights):
    # the filtration of the graph by weights 
    Filtration = returnFiltration(G, weights) 
    # the array that will contain the Betti numbers and Euler characteristic of each element of the filtration 
    betti_array = np.asarray([[0 for i in range(len(weights))] for j in range(4)])
    # We will compute the Betti numbers and Euler characteristic for each graph in the filtration 
    for i in range(len(Filtration)):
        # the Hochschild homology of the current element of the filtration 
        curr_homology = Hochschild(Filtration[i])
        # set the elements of the ith row of betti_array to be the homology of the current element of the filtration
        betti_array[:,i] = curr_homology 
       
    return betti_array        


"""
A function that computes  Betti numbers of the directed flag complex of a graph G in dimensions 0, 1, 2 as well as the Euler characteristic
 e(G),  where e(G) H_{0}(Fl(G)) - H_{1}(Fl(G)) + H_{2}(Fl(G)), where Fl(G)
is the flag complex and H_{i} is ordinary singular homology 
"""
def Standard(G):
    
    #  convert G from a weighted graph to a numpy array and then applying pyflagser to compute homology 
    matrix = nx.to_numpy_array(G)
    homology = pyflagser.flagser_unweighted(matrix, min_dimension=0, max_dimension=2, directed = True, coeff = 2)
    # The array to be returned. 
    betti_array = [0, 0, 0, 0]
    # We set it to be the form [H_{0}^{hoch}(Fl(G)), H_{1}^{hoch}(Fl(G)), H_{2}^{hoch}(Fl(G)), e(G) ]
    betti_array[0:len(homology["betti"])] = homology["betti"]
    betti_array[3] = homology["euler"]
    
    return betti_array

"""
For a digraph G and a collection of weights, the graph will compute the filtration given by the collection of weights
 and return a 4 * len(weights) array containing the Betti numbers and Euler characteristic of each of the elements of the filtration. 
"""
def PersistentStandard(G, weights):
    # the filtration of the graph by weights 
    Filtration = returnFiltration(G, weights) 
    # the array the will contain the Betti numbers and Euler characteristic of each element of the filtration 
    betti_array = np.asarray([[0 for i in range(len(weights))] for j in range(4)])
    # We will compute the Betti numbers and Euler characteristic for each graph in the filtration 
    for i in range(len(Filtration)):
        # the standard homology of the current element of the filtration 
        curr_homology = Standard(Filtration[i])
        # set the elements of the ith row of betti_array to be the homology of the current element of the filtration
        betti_array[:,i] = curr_homology
      
    return betti_array          
     

"""
 a function that will plot the H1 and H2 Hochschild and Standard Betti curves of a graph. 
 G = the graph 
 filetitle = the title of the png file containing the plot
 params = list some properties of the graph to appear on the image's title 
"""
def GraphBettiCurve(G, filetitle, params):     
    # the weights of the graph 
    weights = get_weights(G)
    # the Betti curve of the persistent Hochschild diagram
    hochschild = PersistentHochschild(G, weights) 
    standard = PersistentStandard(G, weights)
    H1_hochschild = hochschild[1]
    H2_hochschild = hochschild[2]
    H1_standard= standard[1]
    H2_standard = standard[2]
    
    # create a plot of the H1 and H2 Betti curves 
    figure, axis = plt.subplots(2, 2)
    
    # the plot is divided into 4 equal sized figures 

    #  H1 standard 
    axis[0, 0].plot(weights, H1_standard)
    axis[0, 0].set_title("H1 Standard")

    # H1 Hochschild
    axis[0, 1].plot(weights, H1_hochschild)
    axis[0, 1].set_title("H1 Hochschild")

    # H2 standard
    axis[1, 0].plot(weights,H2_standard)
    axis[1, 0].set_title("H2 Standard")

    #H2 hochschild 
    axis[1, 1].plot(weights, H2_hochschild)
    axis[1, 1].set_title("H2 Hochschild")
    
    # title figure
    figure.suptitle("Standard vs. Hochschild  " + params, fontsize=12)
    
    # Combine the graphs of each plot and display
    plt.tight_layout()
    plt.savefig(filetitle)
    
    plt.show()
