"""
This file contains the experiment to classify seizures via persistent homology. The input data consists of graphs representing the 
whose vertices represent regions of the brain and whose edges represent CCM correlations between time series of brain activity.
We use the persistent homology pipeline from the file PersistentClassificationExperiment to classify Parkinsons and control patients
"""

# import the basic packages
import itertools
import numpy as np
import networkx as nx
import statistics 
import pandas as pd
import math



# import the classification pipeline 
import PersistentClassificationExperiment as pce

def main():
    # the array containing the graphs representing brain activity 
    samples = []
    # the array containing the classification (seizure or non-seizure) of the graph 
    classification = [] 
    # the arrays containing the patients for which we have seizure data (some patients such as
    #  patient 10 only has seizures lasting < 30 s so were not included in the analysis). 
    # those with label_1 had one folder of seizure data those with label_2 had two folders 
    labels_1 = [1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16]
    labels_2 = [4, 14]
    # read the files containing the average preictal, ictal, baseline for the recordings of each patients as numpy arrays
    # then convert to graphs. 
    for l in labels_1: 
        print("now extracting patient #" + str(l))
        baseline = np.loadtxt("patient_data/data_" + str(l) + "-baseline", delimiter=',')
        preictal = np.loadtxt("patient_data/data_" + str(l) + "-preictal", delimiter=',')
        ictal = np.loadtxt("patient_data/data_" + str(l) + "-ictal", delimiter=',')
        # we subtract the baseline from the preictal and ictal segments to normalize the data 
        preictal_norm = preictal-baseline
        ictal_norm = ictal-baseline
        
        # convert the resulting numpy arrays to weighted graphs 
        G1 = nx.from_numpy_array(preictal_norm, parallel_edges=False, create_using=nx.DiGraph)
        G2 = nx.from_numpy_array(ictal_norm, parallel_edges=False, create_using=nx.DiGraph)
        samples.append(G1)
        samples.append(G2)
    
    # now we go to the patients contained in multiple folders 
    for l in labels_2: 
        print("now extracting patient #" + str(l))
        baseline1 = np.loadtxt("patient_data/data_" + str(l) + "a-baseline", delimiter=',')
        preictal1 = np.loadtxt("patient_data/data_" + str(l) + "a-preictal", delimiter=',')
        ictal1 = np.loadtxt("patient_data/data_" + str(l) + "a-ictal", delimiter=',')
        baseline2 = np.loadtxt("patient_data/data_" + str(l) + "b-baseline", delimiter=',')
        preictal2 = np.loadtxt("patient_data/data_" + str(l) + "b-preictal", delimiter=',')
        ictal2 = np.loadtxt("patient_data/data_" + str(l) + "b-ictal", delimiter=',')

        # we average the precictal and ictal segments from the two files and subtract taht averaged baseline
        preictal_norm = (preictal1 + preictal2 -baseline1 - baseline2) * 0.5
        ictal_norm = (ictal1 + ictal2 -baseline1 - baseline2) * 0.5
    
        # now convert to graphs 
        G1 = nx.from_numpy_array(preictal_norm, parallel_edges=False, create_using=nx.DiGraph)
        G2 = nx.from_numpy_array(ictal_norm, parallel_edges=False, create_using=nx.DiGraph)
        samples.append(G1)
        samples.append(G2)
     # add the classification for each graph ictal = 1, preictal = 0 
     # note that the preictal and ictal segments are added consecutively
    classification = [0] * (2 * (len(labels_1) + len(labels_2)))
    for i in range((len(labels_1) + len(labels_2))):
         classification[2*i] = 1

    """
    Here are the hyperparameters for the experiment. Note that in this experiment the matrices no longer have entries that are correolation values. 
    The entries are in fact differences of correlations. We chose intervals of 0.05, starting with -0.4 since this was close to the minimum weight amongst all graphs
    

    The choice to delete edges with weights above 0 was for two reasons. The computation for the bounds for standard persistent homology was very expensive.
    The problem was that a few of the graphs had too many edges (10000) to reasonably compute all steps in the filtration. But by deleting edges with weights above 0.  
    these graphs shrunk to a more manageable size (5000 vertices) that allowed us to complete the analysis quickly, while not much affecting the results for Hochschild 
    Also there were some graphs which had no weight above 0.05. 
    """
    # delete edges below these threshhold values in the experiment 
    #threshholds_lower = [-0.4,-0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05]
    # the choice of folds for cross-validation 
    #cv_folds = [2, 3, 5] 
    threshholds_lower = [-0.05]
    cv_folds = [5]
   
    # complete the experiment for persistent Hochschild homology 
    pce.MultiClassificationExperiment(samples, classification, 10, 'h', "EPILEPSY", cv_folds, threshholds_lower, [0.0]) 
    # complete the experiment for persistent homology 
    pce.MultiClassificationExperiment(samples, classification, 10, 's', "EPSILEPSY", cv_folds, threshholds_lower, [0.0]) 


if __name__=="__main__":
    main()

