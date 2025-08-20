This repository contains the experiments for an in-progress paper called "Persistent Reachability in Machine Learning". The three experiments are as follows:

1. An experiment which tests the accuracy of using persistent (Hochschild) homology features in an SVM to classify seizure vs. pre-seizure brain states in epileptic patients
  The folder patient_data contains the CCM correlation matrices of both the average seizure (ictal) and pre-seizure (preictal) brain states of the patients. This experiment is completed by
  SEIZURE_EXPERIMENT.py.  
2. An experiment that studies the behavior of the persistent (Hochschild) homology of random graphs. The first and second experiments respectively generate random Erdos-Renyi and preferential graphs with       random integral weights and then the graphs their (H1, H2) betti curves. The third experiment produces plots of the average Betti numbers of Erdos-Renyi graphs with n=100 vertices as the edge-probability    varies. This experiment can be completed by running "RandomGraphExperiments.py"
3. The third experiment tests the accuracy of using persistent (Hochschild) homology features in an SVM to classify Parkinsons and control patients from CCM correlation matrices obtained from MEG time           series. The code to run this experiment is found in "ParkinsonsExperiment.py". However, the correlation matrices are contained in CCM_All.h5 which is proprietary and thus not included in this                repository. So in practice a user cannot run this particular experiment. 


The repository also contains a file called Extract_IEEG_data.py which allows one to compute the CCM correlations of the seizure patients. In order to do this, save the "short time series" files from 
http://ieeg-swez.ethz.ch/ to the same folder as Extract_IEEG_data.py and run it. 
