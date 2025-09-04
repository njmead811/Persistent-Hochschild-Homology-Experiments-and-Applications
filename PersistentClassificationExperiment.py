
"""
The purpose of this file is to test the efficiency of using persistent homology vs persistent Hochschild homology as features 
in a binary graph classification problem. We will use H_{0}, H_{1}, H_{2}

The procedure is as follows: 
    
1. Identify the bounds for which each homology group is nonzero. This step is essential because experimental evidence with random
 graphs showed that the Betti curves for Hochschild homology of random weighted graphs are nonzero only in a very small range of weights.
Also, because the number of graphs in a classification problem is usually small, we will usually only be able to use a small 
number of filtration steps. 

2. Compute the persistent (Hochschild) homology of each graph for an appropriate filtration. Since there are a small number of graphs, there will usually be 
a small number of steps in the filtration.

3. Train an SVM on the classification problem using the homology features for each graph obtained in step 2.    

"""

# first import basic packages such as statistics, numpy etc.  
import itertools
import numpy as np
import networkx as nx
import statistics 
import pandas as pd
import math

# we import the packages related to SVMs and cross-validation (to test accuracy) from scikit learn
from sklearn import preprocessing
from sklearn import svm 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score

# we import the function that computes persistent (Hochschild) homology 
import PersistentHochschild as ph

"""
Given a weighted digraph G, we find all of its weights and compute the associated filtration 
We compute the minimum and maximum weights w^{i}, v^{i} such that H_{i}(G_{w}) is nonzero, where
G_{w} is the subgraph of G consisting of all edges with weights <= w 


The function takes two parameters the graph (G) and homology_type= s, h which is the type of homology
for which we want to compute the bounds for. 

It returns ((w^{0}, v^{0}), (w^{1}, v^{1}), (w^{2}, v^{2}) ) 
"""

def Bounds(G, homology_type):
    
    
    # we return the weights of G and compute the associated filtration 
    weights = ph.get_weights(G)
    Filtration = ph.returnFiltration(G, weights) 
    
    # the minimum weights for which H_{0}, H_{1}, H_{2} are nonzero 
    # it is assumed that the weights of the graph are all < 1. Thus, they are initialized to be 1
    min_0 = 1
    min_1 = 1
    min_2 = 1
    
    # here we find the minimum value in the filtration for which H_{0}, H_{1}, H_{2} are nonzero.
    # we check the graphs in increasing order, starting with the first (smalllest) graph 
    #in the filtration. 
    for i in range(len(Filtration)):
        
        # if we have found a weight for which each of the homologies are non-zero break
        if min_0 != 1 and min_1 != 1 and min_2 != 1:
            break
        # otherwise test the homology of the current graph in the filtration 
        else:
            # the homology of the current graph. If homology_type = h, compute Hochschild, if s compute standard
            hom = 0 
            if homology_type == 'h':
                hom = ph.Hochschild(Filtration[i])
            elif homology_type == 's':
                hom = ph.Standard(Filtration[i])
            # if we have not yet found the minimum bound for homology in a given dimension, and the current homology is nontrivial
            # record the minimum bound
            
            if min_0 == 1:
                if hom[0] != 1:
                    min_0 = weights[i]
            if min_1 == 1:
                if hom[1] != 0:
                    min_1 = weights[i]
            if min_2 == 1:
                if hom[2] != 0:
                    min_2 = weights[i]
                    
    # the maximum weights for which each homology is non-zero.         
    # It is assumed that the weights in the graph are all > -1. Thus, we initialize the max values at -1      
    max_0 = -1
    max_1 = -1
    max_2 = -1
    # here we find the maximum value in the filtration for which H_{0}, H_{1}, H_{2} are nonzero.
    # we check the graphs in descreasing order, starting with the last (largest) graph 
    #in the filtration. 
    for i in range(len(Filtration)-1, -1, -1):
        
        if max_0 != -1 and max_1 != -1 and max_2 != -1:
            break
        else: 
            hom = 0 
            if homology_type == 'h':
                hom = ph.Hochschild(Filtration[i])
            elif homology_type == 's':
                hom = ph.Standard(Filtration[i])
            

            # if we have not yet found the maximum bound for homology being nonzero in a given dimension, and the current homology is nontrivial
            # record the maximum bound       
            if max_0 == -1:
                if hom[0] != 1:
                    max_0 =  weights[i]
            if max_1 == -1:
                if hom[1] != 0:
                    max_1 =  weights[i]
            if max_2 == -1:
                if hom[2] != 0:
                    max_2 =  weights[i]
                    
            
    # return the bounds for which homology is nonzero    
    return ((min_0, max_0), (min_1, max_1), (min_2, max_2) )
    
"""

For i = 0, 1, 2 and a collection Graphs of weighted digraphs, this function will compute the largest (smallest) w^{i} (v^{i})
such that H_{i}(G_{w^{i}}) (H_{i}(G(v^{i}))) is nonzero. 

This function takes two parameters. Graphs, an array of weighted digraphs, and homology_type, if homology_type == 'h' use Hochschild homology
if homology_type == 's' use standard homology. 
"""    
        
 
def BoundsGraphArray(Graphs, homology_type):
    
    # the minimum and maximum weights for which H_{i}(G_{w_{i}}) is nonzero for some G in Graphs
    # since the graphs are all assumed to have -1 < weights < 1, we initialize the maximum (minimum) value to -1 (1)
    min_0 = 1
    max_0 = -1
    min_1 = 1
    max_1 = -1
    min_2 = 1
    max_2 = -1
    
    # these arrays contain for each G in Graphs, the maximum/minimum weights for which H_{i}(G_{w_{i}}) is nonzero 
    arr_min_0 = []
    arr_min_1 = []
    arr_min_2 = []
    arr_max_0 = []
    arr_max_1 = []
    arr_max_2 = []
    
    # we compute the bounds on homology for each graph in Graphs
    for G in Graphs:
        print("Curr_Graph")
        print(G)
        # the bounds for the current graph 
        curr_bounds = Bounds(G, homology_type)
        # if the maximum weight is -1, this means that there was no non-trivial homology in dimension i 
        # otherwise, we add the minimum and maximum weights for which the H_{i}(G_{w}) is nonzero
        # to the array arr_min_i, arr_max_i
        if curr_bounds[0][1] != -1:
            arr_min_0.append(curr_bounds[0][0])
            arr_max_0.append(curr_bounds[0][1])
        if curr_bounds[1][1] != -1:
            arr_min_1.append(curr_bounds[1][0])
            arr_max_1.append(curr_bounds[1][1])
        if curr_bounds[2][1] != -1:
            arr_min_2.append(curr_bounds[2][0])
            arr_max_2.append(curr_bounds[2][1])
    # if there is a graph with non-trivial homology in dimension i, we compute the smallest (largest) weight with nontrivial homology
    # by taking the minimum (maximum) element of the bounds for each Graph. 
    if len(arr_min_0) > 0:
       
        min_0 = min(arr_min_0)
        max_0 = max(arr_max_0)
    if len(arr_min_1) > 0:
         min_1 = min(arr_min_1)
         max_1 = max(arr_max_1)
    if len(arr_min_2) > 0:
         min_2 = min(arr_min_2)
         max_2 = max(arr_max_2)     
    # return the bounds 
    return ((min_0, max_0), (min_1, max_1), (min_2, max_2)  )


"""
For a specified binary classification problem, this function will 

1. use cross validation to test the accuracy of using an SVM with an rbf kernel
2. use combined cross-validation and feature ranking for an SVM with linear kernel 

data = the data
classes = the classes 
cv_folds = number of folds for cross_validation
"""
def custom_cross_val(data, classes, cv_folds):
    # we normalize the data
    X_scaled = StandardScaler().fit_transform(data)
    # we train an SVM with RBF kernel on the normalized vectors to predict the class of each data point and then record the accuracy using cross-validation 
    rbf_clf = svm.SVC(kernel='rbf', random_state=52)
    rbf_cv = cross_validate(rbf_clf, X_scaled, classes, cv=cv_folds, scoring=['accuracy','precision','recall'])
    rbf_scores = [statistics.mean(rbf_cv['test_accuracy']), statistics.stdev(rbf_cv['test_accuracy']), statistics.mean(rbf_cv['test_precision']), statistics.stdev(rbf_cv['test_precision']), statistics.mean(rbf_cv['test_recall']), statistics.stdev(rbf_cv['test_recall'])]
    
    
    # we train another SVM with linear kernel to predict the class of data point then record the accuracy using cross-validation with feature selections
    # We record the mean accuracies as well as the features selected 
    clf = svm.SVC(kernel='linear',random_state=52)
    selector = RFECV(clf, step=1, cv=cv_folds)
    selector = selector.fit(X_scaled, classes)
    fs = np.asarray( [ int(selector.support_[i]) for i in range(len(selector.support_))])
    Linear_Results = [fs, selector.cv_results_["mean_test_score"], selector.cv_results_["std_test_score"]]
    
    return (rbf_scores, Linear_Results)

"""
This is the a single experiment for binary classification of graphs via persistent (Hochschild) homology features. 
The experiment is as follows.

1. Compute the persistent homology of an appropriate filtration based on specified bounds for each graph. Save data to file
2. Use cross-validation to test the accuracy of an SVM trained using the persistent homology as a feature. We test 
   using the raw Betti numbers at each step of the filtration vs the iterated integral. For both we use a linear kernel with feature ranking 
   and an rbf kernel. 

Function parameters : 
    Graphs - a weighted array of digraphs,
    classes - an array that's ith element is the class of the ith element of Graphs'
    bounds - the bounds used to compute h_0, h_1 and h_2 features as a 3 times 2 array
    num_filt_steps - the number of steps in the filtration in persistent homology
    homology type - if 's' compute standard persistent homology if 'h' compute hochschild homology
    cv_folds - number of folds for cross-validation.
    experiment_name - the type of data in the experiment/name of experiment 
    experiment_parameters - additional information about the experiment parameters
    
"""


def GraphClassificationExperiment(Graphs, classes, bounds, num_filt_steps, homology_type, cv_folds, experiment_name, experiment_parameters):
    
    print(bounds)
    
    # This vector will be a concatenation of the vectors contain the raw Betti numbers (H_{0}, H_{1}, H_{2}) at each step in the filtration for each graph. 
    Betti_Data = np.zeros((len(Graphs), num_filt_steps+1))
    
    # This vector will be a concatentation of the vectors containing the iterated integral of the betti curve  (H_{0}, H_{1}, H_{2}) at each step in the filtration 
    Integral_Data = np.zeros((len(Graphs), num_filt_steps))
    
    # the weights for the filtration that we compute persistent H0 for
    weights_h0 = [bounds[0][0] + (bounds[0][1] - bounds[0][0] ) * i/num_filt_steps for i in range(num_filt_steps+1)] 
    
    
    for i in range(len(Graphs)):
        # compute the persistent homology for the set of weights. If homology_type = 'h' compute hochschild. If homology_type = 's' compute standard
        Persistent_H0 = 0
        if homology_type == 'h':
            Persistent_H0 = ph.PersistentHochschild(Graphs[i], weights_h0)[0]
        elif homology_type == 's':
            Persistent_H0 = ph.PersistentStandard(Graphs[i], weights_h0)[0]
        # if the homology type is not specified appropriately then throw an error
        else:
            raise ValueError("Homology type not specified correctly")
        # compute the iterated integral of the Betti curve at each step of the filtration.     
        Integral_Data[i][0] = (weights_h0[1]-weights_h0[0]) * (Persistent_H0[0] + Persistent_H0[1])/2 
        for j in range(1, num_filt_steps):
            Integral_Data[i][j] = (weights_h0[j+1]-weights_h0[j]) * (Persistent_H0[j] + Persistent_H0[j+1])/2 + Integral_Data[i][j-1]
        # Store the raw betti numbers 
        Betti_Data[i] = Persistent_H0
    
   

    #If there is a graph with nontrivial H1, H2 at some step of the filtration, add
    # data about persistent H1, H2 to the Integral_Data and Betti_Data vector. 
    # Recall that if there is no non-trivial homology in dimension i bounds[i][1] will remain at the default value -1
    for d in [1, 2]: 
        # By the construction fo the bounds function this is equivalent to there being some non-zero homology in dimension d at some step of the 
        # filtration 
        if bounds[d][1] != -1:
            # arrays that contain the raw dth Betti numbers 
            Hd_bettis = np.zeros((len(Graphs), num_filt_steps+1))
            # array that contains the Betti integral for Hd 
            Hd_cols = np.zeros((len(Graphs), num_filt_steps))
            # the weights in the filtration for persistent homology
            weights_hd = [bounds[d][0] + (bounds[d][1] - bounds[d][0] ) * i/num_filt_steps for i in range(num_filt_steps +1)]
        
            for i in range(len(Graphs)):
                # compute the persistent homology for the set of weights. If we homology_type = 'h' compute Hochschild, if homology_type = 's' compute standard
                Persistent_Hd = 0
                if homology_type == 'h':
                    Persistent_Hd = ph.PersistentHochschild(Graphs[i], weights_hd)[d]
                elif homology_type == 's':
                    Persistent_Hd = ph.PersistentStandard(Graphs[i], weights_hd)[d]
            
            
                # compute the iterated integral of the Betti curve at each step of the filtration.     
                Hd_cols[i][0] = (weights_hd[1]-weights_hd[0]) * (Persistent_Hd[0] + Persistent_Hd[1])/2 
                for j in range(1, num_filt_steps):
                    Hd_cols[i][j] = (weights_hd[j+1]-weights_hd[j]) * (Persistent_Hd[j] + Persistent_Hd[j+1])/2 + Hd_cols[i][j-1]
                Hd_bettis[i] = Persistent_Hd
            Betti_Data = np.concatenate(((Betti_Data), Hd_bettis), axis=1)
            Integral_Data = np.concatenate((Integral_Data, Hd_cols), axis=1)
    
     
   
    
    """
    Here we save all of the raw Betti numbers of each graph, as well as its class to a file 
    """
    class_col = np.zeros((len(classes), 1))
    for i in range(len(classes)):
        class_col[i] = classes[i]    
    
    Curr_Data = np.concatenate((class_col, Betti_Data), axis=1)
    
    # a string representing the type of homology we are computing - can be either Hochschild or Standard
    homology_name = ""
    if homology_type == 's':
        homology_name = "STANDARD"
    elif homology_type == 'h':
        homology_name = "HOCHSCHILD"
    np.savetxt(homology_name + "-" + experiment_name +"-data-" + experiment_parameters + ".csv", Curr_Data, delimiter=",")

    
    
    # we use our custom cross-validation function to test the accuracy of using both the raw Betti numbers and the Betti curve to classify our graphs
    rbf_scores, Linear_Results = custom_cross_val(Integral_Data, classes, cv_folds)
    rbf_scores_raw, Linear_Results_Raw = custom_cross_val(Betti_Data, classes, cv_folds)

    return (Linear_Results, rbf_scores, Linear_Results_Raw, rbf_scores_raw)
  
"""
The purpose of this function is to repeat the classification experiment for pairs of specified threshhold values and choices of cross validation folds.
The results of each experiment will be saved to a file. 

We will add an additional step of preprocessing before each experiment where we remove the edges above and below a certain threshhold. 

The following inputs are constant for each experiment  

Graphs - a weighted array of graphs
classes - an array containing the classes (binary 0 or 1) assigned to each graph
num_filtration_steps - the number of steps in the filtration to compute 
homology_type - type of homology to compute (i.e. Hochschild or standard) 
experiment_name - information about the experiment/data that the method was tested on. 
The following are variable 

thresh_lower_set - the set of lower threshholds
thresh_upper_set - the set of upper threshholds 
cv_folds - the set of choices for the number of folds 
"""
def MultiClassificationExperiment (Graphs, classes, num_filt_steps, homology_type, experiment_name, cv_folds, thresh_lower_set, thresh_upper_set): 
    
    # the set of the bounds on homology, stored as a multidimensional array
    all_bounds = np.zeros((len(thresh_lower_set), len(thresh_upper_set), 6 ))
    # the results of the graph classification for the rbf kernel for both iterated integral and raw betti
    all_rbf_scores = np.zeros((len(thresh_lower_set), len(thresh_upper_set), len(cv_folds) , 6 ))
    all_rbf_scores_raw = np.zeros((len(thresh_lower_set) , len(thresh_upper_set), len(cv_folds), 6 ))
   
    # The results of the classification by SVM with linear kernel for both the raw betti and iterated integral will be saved in a file
    # whose content is given by the following strings 
    Linear_Results_Txt = ""
    Linear_Results_Raw_Txt = ""
    
    for i in range(len(thresh_lower_set)):
        for j in range(len(thresh_upper_set)):
           
            # remove the edges of the graphs which have weights above and below a certain value 
            # we perform this step now, since the choice of cross-validation folds does not affect the graphs, and we do not want to unnecessarily 
            # recompute the threshholded values for the graphs and the bounds, since the latter is very computationally intensive. 
            GraphsTh = [ph.Threshhold(Graphs[l], thresh_lower_set[i], thresh_upper_set[j]) for l in range(len(Graphs))]
            
             print("We are currently computing the bounds for threshhold choice " + str(i*len(thresh_upper_set) + j))
            # compute the bounds for the filtration used to compute persistent homology features. 
            bounds = BoundsGraphArray(GraphsTh, homology_type)
            
            
            # record the bounds for the current experiment.
            all_bounds[i][j] = np.asarray(bounds).reshape(-1)

            for k in range(len(cv_folds)):
                    print("this is experiment #" + str(i * len(thresh_upper_set) * len(cv_folds) + j* len(cv_folds) + k))
                    # compute the results for the current experiment
                    Linear_Results, rbf_scores, Linear_Results_Raw, rbf_scores_raw = GraphClassificationExperiment( GraphsTh, classes, bounds,  10, homology_type, cv_folds[k], "PARKINSONS", "lower_thresh=" + str(thresh_lower_set[i]) + "upper_thresh=" + str(thresh_upper_set[j]))
                    # add the rbf scores of the current experiment to the arrays of rbf scores
                    all_rbf_scores[i][j][k] = rbf_scores
                    all_rbf_scores_raw[i][j][k] = rbf_scores_raw
                    
                    # add information about the results of the linear kernel SVM to the strings Linear_Results_Raw_Txt
                    fs_raw_curr = Linear_Results_Raw[0]
                    mean_raw_curr = Linear_Results_Raw[1]
                    std_raw_curr = Linear_Results_Raw[2]
                    # the report about the linear kernel SVM in the current experiment
                    Curr_Report_RAW = "CV_FOLDS = " + str(cv_folds[k]) + "THRESHHOLD_LOWER = " + str(thresh_lower_set[i]) + " THRESHHOLD_UPPER = " + str(thresh_upper_set[j]) + "\n" 
                    Curr_Report_RAW = Curr_Report_RAW + " ".join(str(x) for x in fs_raw_curr) + "\n" + " ".join(str(x) for x in mean_raw_curr) + "\n" + " ".join(str(x) for x in std_raw_curr) + "\n"
                    # add the information about the current experiment to the Linear_Result text file 
                    Linear_Results_Raw_Txt = Linear_Results_Raw_Txt + Curr_Report_RAW
                    
                    # repeat the same thing for the results of the linear kernel SVM for the Betti curve classification 
                    fs_curr = Linear_Results[0]
                    mean_curr = Linear_Results[1]
                    std_curr = Linear_Results[2]
                    Curr_Report = "CV_FOLDS = " + str(cv_folds[k]) + "THRESHHOLD_LOWER = " + str(thresh_lower_set[i]) + "THRESHHOLD_UPPER = " + str(thresh_upper_set[j]) + "\n" 
                    Curr_Report = Curr_Report + " ".join(str(x) for x in fs_curr) + "\n" + " ".join(str(x) for x in mean_curr) + "\n" + " ".join(str(x) for x in std_curr) + "\n"
                    Linear_Results_Txt = Linear_Results_Txt + Curr_Report
    # Now we save the results of the experiment for different choices of kernel, etc. to a file. The results of the classification with RBF kernel will be saved as a 2-dimensional matrix 
    # same as the bounds, first we reshape the bounds to fit into a 2 dimensional array. Then we put into a dataframe 
    all_bounds_2D = all_bounds.reshape((len(thresh_lower_set) * len(thresh_upper_set) * len(cv_folds), 6) )
    print(all_bounds_2D)
    bounds_df = pd.DataFrame(all_bounds_2D, columns=["H0_min", "H0_max", "H1_min", "H1_max", "H2_min", "H2_max"])        
    
    # reshape the array containing the scores for the Betti curve rbf kernel classification, and put in dataframe form. 
    all_rbf_scores_2D = all_rbf_scores.reshape((len(thresh_lower_set) * len(thresh_upper_set) * len(cv_folds), 6))
    rbf_df = pd.DataFrame(all_rbf_scores_2D, columns=["acc_mean", "acc_std", "prec_mean", "prec_std", "rec_mean", "rec_std"])
    
    # reshape the array containing the scores for the raw Betti number rbf kernel classification, and put in dataframe form. 
    all_rbf_scores_raw_2D= all_rbf_scores_raw.reshape((len(thresh_lower_set) * len(thresh_upper_set) * len(cv_folds), 6))
    raw_rbf_df = pd.DataFrame(all_rbf_scores_raw_2D, columns=["acc_mean", "acc_std", "prec_mean", "prec_std", "rec_mean", "rec_std"])
    
    
    
    # save the dataframes containing information about the bounds, SVM classification with both RBF and Linear Kernels
    
    # the type of homology used 
    homology_name = ""
    if homology_type == 's':
        homology_name = "STANDARD"
    elif homology_type == 'h':
        homology_name = "HOCHSCHILD"
    
    bounds_df.to_csv(homology_name +'_BOUNDS_' + experiment_name + '.csv')
    rbf_df.to_csv(homology_name+ "_RBF_SCORES_" + experiment_name + ".csv")
    raw_rbf_df.to_csv(homology_name + "_RBF_RAW_SCORES_" + experiment_name + ".csv")
    # save the results of the linear classification to text files. 
    with open(homology_name + "_LINEAR_SCORES_" + experiment_name + ".txt", "w") as f:
        f.write(Linear_Results_Txt)
    with open(homology_name + "_LINEAR_RAW_SCORES_" + experiment_name + ".txt", "w") as f:
         f.write(Linear_Results_Raw_Txt)         
                


