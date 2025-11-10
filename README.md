This repository contains the experiments for an in-progress paper called "Persistent Reachability Homology in Machine Learning Applications." It is available on arxiv at https://arxiv.org/abs/2511.04825. 

The two experiments are as follows:

1. An experiment which tests the accuracy of using persistent (Hochschild) homology features in an SVM to classify seizure vs. pre-seizure brain states in epileptic patients. The folder patient_data contains the CCM correlation matrices of both the average seizure (ictal) and pre-seizure (preictal) brain states of the patients. This experiment is completed by running SEIZURE_EXPERIMENT.py.  
2. An experiment that studies the behavior of the persistent (Hochschild) homology of random graphs. Firstly, the experiment will generate a number of random Erdos-Renyi and preferential attachment graphs with random integral weights and then  saves their (H1, H2) betti curves as a png file. Then it produces plots of the average Betti numbers of Erdos-Renyi graphs with n=100 vertices as the edge-probability    varies. This experiment can be completed by running "RandomGraphExperiments.py"



The repository also contains a file called Extract_IEEG_data.py which allows one to compute the CCM correlations of the seizure patients. In order to do this, save the "short time series" files from 
http://ieeg-swez.ethz.ch/ to the same folder as Extract_IEEG_data.py and run it. 
