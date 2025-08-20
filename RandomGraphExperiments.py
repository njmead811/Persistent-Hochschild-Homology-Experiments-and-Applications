"""
This file contains two experiments for the behavior of (persistent) Hochschild homology for random graphs

In the first experiment, we will generate random graphs based on the Erdos-Renyi and preferential attachment models, and then we 
will graph their persistent homology (both standard and hochschild)

In the second experiment we will record the average Betti numbers (ordinary and Hochschild), for various hyperparameters 
for the Erdos-Renyi and graph them. 
"""

# import the necessary basic packages 
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np 
import os 
import networkx as nx


# import the files that contain tools for generating random graphs and persistent homology. 
import PersistentHochschild as ph
import GraphFunctions as gf 

"""
This function will generate graphs of the Betti curves of a random graph generated using the Erdos-Renyi model
and then save them to a file 

num_trials - the number of graphs to generate for each p-value
num_vertices - the number of vertices of the graphs
min_weight, max_weight - the maximum and minimum weights of the edges
p_values - the set of edge probabilities for the graphs
curr_directory - the directory to save the files  
"""
def ExperimentRandomER(num_trials, num_vertices, min_weight, max_weight, p, curr_directory):
    print("PH Experiment with Erdos-Renyi Graphs")
    # if the curr_directory doesn't exist, then create it
    if not os.path.isdir(curr_directory):
        os.mkdir(curr_directory)
    for i in range(len(p)):
        for j in range(num_trials):
            print("now completing trial" + str((i * num_trials) + j) )
            # title of the current image
            curr_title = "n= " + str(num_vertices) + " p=" + str(p[i]) + "trial" + str(j) + '.png'
            # generate the random graph and then visualize using the function GraphBettiCurve. 
            G = gf.getWeightedErdosRenyiGraph(num_vertices, p[i], min_weight, max_weight)
            ph.GraphBettiCurve(G, curr_directory + "/" + curr_title, "Density = " + str(p[i]))  
"""
This function will generate graphs of the Betti curves of random graph generated using the preferential attachment model

num_trials - the number of graphs to generate for each p-value
num_vertices - the number of vertices of the graphs
min_weight, max_weight - the maximum and minimum weights of the edges
m , delta - the set of hyperparameters for the random attachment graph  
curr_directory - the directory to save the files in 
"""
def ExperimentRandomPE(num_trials, num_vertices, min_weight, max_weight, delta, m, curr_directory):
    print("PH Experiment with Preferential Attachment Graphs")
    # if the curr_directory doesn't exist, then create it
    if not os.path.isdir(curr_directory):
        os.mkdir(curr_directory)
    # 
    # we now generate the random graph 
    for i in range(len(delta)):
        for j in range(len(m)):
            for k in range(num_trials):
                print("now completing trial" + str((i * j * num_trials) + k) )
                # the title of the current image 
                curr_title = "n= " + str(num_vertices) + " delta=" + str(delta[i]) + "m=" + str(m[i]) + "trial" + str(j) + '.png'
                # generate the random graph and then visualize using the GraphBettiCurve function.
                G = gf.Weighted_Preferential_Attachment_Graph(num_vertices, delta[i], m[j], min_weight, max_weight)
                ph.GraphBettiCurve(G, curr_directory + "/" + curr_title, "delta = " + str(delta[i]) + " m = " + str(m[j]))  

"""
This function will compute the mean and standard deviation of the Betti numbers for both Hochschild and standard homology 
for graphs generated using the Erdos-Renyi model for fixed hyperparameters 

num_graphs = number of random graphs to generate
num_vertices = number of vertices each graph should have
edge_probability=  probability of edge existing in Erdos-Renyi model 
"""

def get_statistics_betti(num_graphs, num_vertices, edge_probability):        
    # the betti numbers hochschild and standard of all the graphs
    All_H1_H = []
    All_H2_H = []
    All_H1_S = []
    All_H2_S = []
    for i in range(num_graphs):
        
        # generate a random graph
        G = nx.erdos_renyi_graph(num_vertices, edge_probability, directed = True)
        
        # compute Hochschild and standard homology of the current graph 
        hochschild = ph.Hochschild(G) 
        standard = ph.Standard(G)
        H1_hochschild = hochschild[1]
        H2_hochschild = hochschild[2]
        H1_standard= standard[1]
        H2_standard = standard[2]
        # add Hochschild and standard to the arrays of Betti numbers
        All_H1_H.append(H1_hochschild)
        All_H2_H.append(H2_hochschild)
        All_H1_S.append(H1_standard)
        All_H2_S.append( H2_standard)
    # compute the means and standard deviations of Hochschild and standard Betti numbers
    mean_H1 = round(statistics.mean(All_H1_H), 2)
    mean_H2 = round(statistics.mean(All_H2_H), 2)
    mean_S1 = round( statistics.mean(All_H1_S), 2)
    mean_S2 = round(statistics.mean(All_H2_S), 2)
    std_H1 =  round(statistics.stdev(All_H1_H), 2)
    std_H2 =  round(statistics.stdev(All_H2_H), 2)
    std_S1 = round(statistics.stdev(All_H1_H), 2)
    std_S2 = round(statistics.stdev(All_H2_H), 2)
    # return the results
    return ((mean_H1, std_H1), (mean_H2, std_H2), (mean_S1, std_S1), (mean_S2, std_S2))


"""
This function will graph the average Betti numbers for Erdos-Renyi graphs with p-values specified in edge_probabilities
"""
def GraphAverageBetti(num_graphs, num_vertices, edge_probabilities):
    print("Experiment with Betti Numbers of Erdos-Renyi Graphs")
    # the mean and standard deviation for the Hochschild and standard Betti numbers for Erdos-Renyi graphs with specified edge probability
    mean_H1_H = []
    mean_H2_H = []
    mean_H1_S = []
    mean_H2_S = []
    std_1_H = []
    std_2_H = []
    std_1_S = []
    std_2_S = []
    
    # compute the mean and standardd deviation for the graphs with various p values
    for p in edge_probabilities:
        print("current p-value" + str(p))
        stats = get_statistics_betti(num_graphs, num_vertices, p)  
        mean_H1_H.append(stats[0][0])
        std_1_H.append(stats[0][1])
        mean_H1_S.append(stats[2][0])
        std_1_S.append(stats[2][1])
        mean_H2_H.append(stats[1][0])
        std_2_H.append(stats[1][1])
        mean_H2_S.append(stats[3][0])
        std_2_S.append(stats[3][1])
    
    # record the results in a dataframe
    d = {'edge_probability': edge_probabilities,  'HH-mean_H1': mean_H1_H, 'HH_std_H1': std_1_H, 'HH-mean_H2': mean_H2_H, 'HH_std_H2': std_2_H, 'S-mean_H1': mean_H1_S,  'S-std_H1': std_1_S, 'S-mean_H2': mean_H2_S, 'S_std_H2': std_2_S}
    df = pd.DataFrame(data=d)
    df.to_csv('Summary_Statistics_Betti.csv')
    

    H1_Hochschild_Mean = df['HH-mean_H1'].tolist()
    H1_Standard_Mean = df['S-mean_H1'].tolist()
    
    # create a plot showing the difference between the 1st betti for Hochschild and standard homology
    figure, axis = plt.subplots(2)
    
    # plot the 1st standard Betti  
    axis[0].plot(edge_probabilities, H1_Standard_Mean)
    axis[0].set_title("H1 - Standard")

    # plot the 1st Hochschild Betti     
    axis[1].plot(edge_probabilities, H1_Hochschild_Mean)
    axis[1].set_title("H1 Hochschild")
    figure.suptitle("Standard vs. Hochschild H1", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Standard vs. Hochschild H1.png")
    
    plt.show()

    H2_Hochschild_Mean = df['HH-mean_H2'].tolist()
    H2_Standard_Mean = df['S-mean_H2'].tolist()
    # create a plot showing the difference between the 2nd Betti for Hochschild and standard homology
    figure2, axis2 = plt.subplots(2)
    
    # plot the 2nd standard Betti  
    axis2[0].plot(edge_probabilities, H2_Standard_Mean)
    axis2[0].set_title("H2 - Standard")

    # plot the 2nd Hochschild Betti     
    axis2[1].plot(edge_probabilities, H2_Hochschild_Mean)
    axis2[1].set_title("H2 Hochschild")
    figure2.suptitle("Standard vs. Hochschild H2", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Standard vs. Hochschild H2.png")
    
    plt.show()
    # return the dataframe containing the information about the standard deviation and mean Betti numbers
    return df


def main():
   # our first experiment is to produce random Erdos-Renyi graphs of 100 vertices and graph their persistent homology

   
   p = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
   
   ExperimentRandomER(3, 100, 1, 100, p, "RandomERGraphs")
   
   # our second experiment is to produce random weighted preferential attachment graphs of 100 vertices and graph their persistent homology
   
   delta = [1, 2, 3, 4]
   m = [1, 2, 3, 4, 5]
   ExperimentRandomPE(3, 100, 1, 100, delta, m, "RandomPEGraphs")
   
   # Our third experiment is to graph the average Betti number for Hochschild and Standard H1 and H2 for Erdos-Renyi graphs with 
   # p = 0.002* i for i = 1-100  
   
   
   q = [0.002 * i for i in range(0, 100)]
   GraphAverageBetti(100, 100, q)

if __name__ == "__main__":
    main()

            
            
