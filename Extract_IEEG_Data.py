'''
The "short-term dataset" from http://ieeg-swez.ethz.ch/ contains time series representing
 100 recordings of seizures from 16 patients. Each recording consists of 3 minutes of preictal segments
 (i.e., immediately before the seizure onset), the ictal segment (ranging from 10s to 1002s), and 
 3 minutes of postictal time (i.e., immediately after seizure ending). 
 
 For every seizure recording, we will extract the time series corresponding to 
 1. first 30s of the recording (baseline segment)
 2. the 1min-30s before seizure onset (preictal segment)
 3. first 30s of the seizure (ictal segment)

The time series for each segment consists of the time series for individual electrodes placed around the
patient's head. For the (30s) ictal, preictal and baseline segments of each recording extracted above we will compute the 
CCM correlation matrix of the electrode time series.

We average the ictal, preictal and baseline CCM correlation matrices of each patient and save them in the current directory. 
'''



# Here we import necessary packages
# Of particular importance is the package for CCM correlation. 
from os.path import dirname, join as pjoin
import scipy.io as sio
from causal_ccm.causal_ccm import ccm
import random
import numpy as np
import os
from os import listdir
from os.path import isfile, join




"""
For a given patient, this function computes the CCM correlation matrices for the ictal, preictal and baselines segments of 
each seizure recording. Then the averages of these matrices are saved in a .txt file, to be used in the later analysis

In the parent directory, there is a folder for each patient labeled 
ID+ patient_id. This contains all of the recordings of the patient's seizures.
"""

def CreateCorrelationMatrix(patient_id):
    print("now extracting data for patient " + patient_id)
    
    # the path of the folder containing the data of the current patient
    patient_path =  "ID" + str(patient_id)
    
    # the list of files each containing a record of a single seizure (for the current patient)
    patient_files = [f for f in listdir(patient_path) if isfile(join(patient_path, f))]
    # print the names of the files
    print("files in directory")
    print(patient_files)
    
    
    
    
    
    
    # change directory to folder containig data on current patient
    os.chdir(os.getcwd() + '/' + patient_path)
    
    # The number of electrodes for the current patient 
    num_electrodes = sio.loadmat( patient_files[0])["EEG"].shape[1]
    
   
    
    # In our analysis, we can only use patient recordings which have seizures longer than 30s. 
    # This variable represents the number of recordings of seizures >= 30s
    num_seizures = 0 
    
    # record the sums of the ictal, preictal and baseline CCM correlation matrices for each recording of a seizure of more than 30s
    ictal_sum = np.zeros((num_electrodes, num_electrodes))
    preictal_sum = np.zeros((num_electrodes, num_electrodes))
    baseline_sum = np.zeros((num_electrodes, num_electrodes))
    
   
    
    
    
    """
     We read each recording of a seizure for the current patient. If the recording has a seizure >= 30s we compute the CCM correlations of the ictal, preictal and 
     baseline segments. 
     
     The recordings consist of a matrix whose columns each represent a time series for a single electrode. The measurrements in the time series for each electrode
     occur once every 1/512 seconds
    """
    for f in patient_files:
        
        """
        We open the current recording 
        """
        print("currently extracting:" + f)
        muf = sio.loadmat( f)
        recording = muf["EEG"]
        
        
        
        # the indices of start and end point of the current seizure 
        start_point = 3 * 60 * 512 
        end_point = recording.shape[0] - 3 * 60 * 512
        
        # we only compute the correlation matrices for recordings of length >= 30s 
        if (end_point - start_point) >= 30 * 512:
            
            # add to the total of recordings with length >= 30s
            num_seizures += 1 
            
            # extract the 30s baseline, ictal and preictal segments of the time series we use in our analysis
            baseline = recording[0:(60 * 512), : ]
            ictal = recording[start_point:(start_point + 30 * 512), : ]
            preictal = recording[2 * 60 * 512: 150 * 512, :]
            
            # the correlation matrices for these segments,; we will compute the entries below
            corr_matrix_baseline = np.zeros((num_electrodes, num_electrodes))
            corr_matrix_ictal = np.zeros((num_electrodes, num_electrodes))
            corr_matrix_preictal = np.zeros((num_electrodes, num_electrodes))
            
            
            """
            The parameters for the CCM corellation. tau is time lag, E is embedding dimension of the manifild and L is time horizon, which
            is related to the percentage of the time series to consider when computing CCM correlation. This L value was chosen to be as large as possible 
            so that the runtime of the program was reasonable. It seemed from brief experiments that L > 100 didn't much change the calculated correlation values much
            """
            tau = 1 
            E = 2 
            L=1000
            
            
            # compute each entry of the corellation matrix
            for coord_1 in range(num_electrodes):
                print(str(coord_1))
                for coord_2 in range(num_electrodes):
                    print("currently calculating " + str(coord_1) + "," + str(coord_2))
                    
                    # compute the correlation between the preictal segments for electrode coord_1 and coord_2. Then print 
                    X_pre = [preictal[i, coord_1] for i in range(preictal.shape[0])]
                    Y_pre = [preictal[i, coord_2] for i in range(preictal.shape[0])]
                    ccm_pre = ccm(X_pre, Y_pre,
                                    tau, E, L)
                    corr_matrix_preictal[coord_1, coord_2] = ccm_pre.causality()[0]
                    print("preictal")
                    print(ccm_pre.causality()[0])
                    # compute the correlation between the ictal segments for electrode coord_1 and coord_2. Then print 
                    X_ictal =  [ictal[i, coord_1] for i in range(ictal.shape[0])]
                    Y_ictal = [ictal[i, coord_2] for i in range(ictal.shape[0])]
                    ccm_ictal = ccm(X_ictal, Y_ictal,
                                    tau, E, L)
                    corr_matrix_ictal[coord_1, coord_2] = ccm_ictal.causality()[0]
                    print("ictal")
                    print(ccm_ictal.causality()[0])
                    # compute the correlation between the baseline segments for electrode coord_1 and coord_2.  Then print 
                    X_baseline =  [baseline[i, coord_1] for i in range(baseline.shape[0])]
                    Y_baseline = [baseline[i, coord_2] for i in range(baseline.shape[0])]
                    ccm_baseline = ccm(X_baseline, Y_baseline,
                                    tau, E, L)
                    corr_matrix_baseline[coord_1, coord_2] = ccm_baseline.causality()[0]
                    print("baseline")
                    print(ccm_baseline.causality()[0])
            """
            Add the current corellation matrices for ictal/preictal/baseline to the summed ictal/preictal/baseline corellation matrices
            """
            preictal_sum = preictal_sum + corr_matrix_preictal
            ictal_sum = ictal_sum + corr_matrix_ictal
            baseline_sum = baseline_sum + corr_matrix_baseline
     
        # if there are seizure recordings >= 30s, compute the average of the ictal/preictal/baseline corellation matrices. Then save each to a seperate file. 
        if num_seizures > 0:
            ictal_avg = ictal_sum / num_seizures
            preictal_avg = preictal_sum / num_seizures
            baseline_avg = baseline_sum / num_seizures
        
            np.savetxt('data_' + str(patient_id) + "-ictal", ictal_avg, delimiter=',')
            np.savetxt('data_' + str(patient_id) + "-preictal", preictal_avg, delimiter=',')
            np.savetxt('data_' + str(patient_id) + "-baseline", baseline_avg, delimiter=',')
        
        
    


def main():   
    labels =  [ "4a", "4b", "1", "2", "3" "13a", "13b", "14a", "14b", "15", "16"] + [str(i) for i in range(5, 13)] 
    for l in labels: 
        CreateCorrelationMatrix(l)

    
    

if __name__=="__main__":
    main()






