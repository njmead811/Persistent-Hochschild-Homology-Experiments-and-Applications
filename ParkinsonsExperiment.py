"""
This file contains the experiment to classify MEG data for both Parkinsons and control patients via persistent homology. 
We use the persistent homology pipeline from the file Persistent_Classification_Experiment to classify Parkinsons and control patients
"""
# import packages
import h5py
import numpy as np
import pandas as pd
import networkx as nx
import pyflagser 
# we import the file that contains the persistent homology pipeline
import PersistentClassificationExperiment as pce


def main():   
    
    
    # read the file containing the MEG data 
    f = h5py.File('CCM_All.h5','r')
    # the data for control subjects (in the form of a dataframe)
    data_control = f['control']
    # the data for Parkinsons subject (in the form of a dataframe)
    data_patient = f['patient']
    
    # this array will contain digraphs representing the controls and the Parkinsons patients 
    samples = []
    # For each control subject add it to the array of samples
    for key in data_control.keys():
        # convert the data for the control subject to a numpy array
        MEG_matrix_control = np.array(data_control[key])
        # then convert to digraph
        G = nx.from_numpy_array(MEG_matrix_control,create_using=nx.DiGraph)
        samples.append(G)
    
    # For each control subject add it to the array of samples
    for key in data_patient.keys():
       MEG_matrix_patient = np.array(data_patient[key])
       samples.append(nx.from_numpy_array(MEG_matrix_patient,create_using=nx.DiGraph))
    
    # this array contains the data for the status of each patient (1 = control 0 = parkinsons)    
    classification = np.asarray([0 for i in range(len(data_control) + len(data_patient))])
    classification[0:len(data_control)] = [1] * len(data_control)
    
    
    """
    We run the experiment for various hyperparameters. 
    The weights of the graph represent correlations between activity in various areas of the brain.
    It makes sense that weights of negative or low correlation will not be relevant to classification so we 
    first delete weights below a certain level from the graphs. 
    
    The experiment allows one to also delete weights above a certain level, which we do not do here - effectively we do this by saying the 
    upper threshhold is 1.0 (no weight is more than 1 since the weights represent correlations)
    
    We do cross-validation with folds = 2, 3, 5
    """
    
    
    threshholds_lower = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 ]
    cv_folds = [2, 3, 5]
    
    
    # complete the experiment for Hochschild homology
    pce.MultiClassificationExperiment(samples, classification, 10, 'h', "PARKINSONS", cv_folds, threshholds_lower, [1.0]) 
    # complete the experiment for standard homology 
    pce.MultiClassificationExperiment(samples, classification, 10, 's', "PARKINSONS", cv_folds, threshholds_lower, [1.0]) 

if __name__=="__main__":
    main()



