# Protein-Ubiquitination-Sites
The goal of this project is to develop a pattern classification system using the given training dataset in order to accurately predict ubiquitination on a blind test dataset. 

## Dataset Description
The dataset (which is not included here due to size restrictions) is a collection of protein windows centered on lysine residues. 80% of the dataset is provided with its correctly  classified  classes:  the  positive  windows  correspond  to  the  sites  that  are  known  to  be ubiquitinated  and  the  negative  windows  are  assumed  to  be  ubiquitinated.  Among  the  dataset, only 30% of the data was labelled and each project team is allowed to request to label up to 1000 unlabelled  sites.  The  20%  of  the  total  dataset  is  withheld  as  a  blind  test  set  to  be  identified  as ubiquitination. Initial features for each site has been selected by the application of ProtDCal and reduced to 435 features. 

## High-Level Solution
We chose to use the multilayer  perceptron  as  a  pattern  classifier  and  genetic  algorithm  as  a  feature  selector.  Genetic algorithm is derived from the natural selection, the process of biological evolution. At each step of the algorithm, it randomly selects individuals  from  the  present  generation  and  produces  the  next  generation.  Over  consecutive generations,  the ‘fittest’survivor among  individuals proceedsto  the  next  generation  and  the successful point in each competition provides a possible solution for solving a problem.

# Method
## 1) Data Pre-processing
Several  preprocessing  steps  needed  to  be  taken  prior  to  conducting  feature  selection  and  the training  of  the  classifier. MATLAB automatically normalizes the inputs of the dataset. The missing values were replaced by the column medians. The mean absolute deviation of values within  each  feature  column  were  calculated.  Values  that  were  more  than  10  deviations  away from  the  median  were  replaced  by  the  median.  This  process  was  used  as  it  is  less  error  prone than standard deviations from the mean. Four methods were examined and compared iteratively during the training and testing of  the  classifiers to handle class imbalance. The  first  technique  was  the  synthetic  minority  oversampling  technique (SMOTE);  it  was  used  on  70%  of  the  labelled  data  to  equalize  the  number of  samples  in  each class. Adaptive SMOTE is second technique which was used; it samples the minority class near the  boundary  between  both  classes.  The  two  final  techniques  explored  include  penalizing the algorithm for misclassifying positive class samples and undersampling of the majority class. The four  methods  discussed  above  were  to  be  evaluated during  the  testing  of  the  classifier. 

## 2) Feature Selection
For the feature  selection  method,  we identified two  methods  to  increase  performance  of  our classifier(in  this  case,  performance  is  the  optimal  precision/recall  point).  First,  we  selected CfsSubsetEval  in  Weka  to  perform  feature  selection.  This  evaluator  considers  the  individual predictive ability of each feature and  evaluates a  possible subset of attributes. The subset of features are correlated with the class. After a time-consuming and computationally heavy greedy search, we generated 60 selected features. Greedy algorithm provides an optimal choice at each stage,  the  chosen  solution  contributes  to  a  solution  and  iterates.  Based  on  the resulting features from greedy search, we implemented a genetic search feature selection method in Weka to boost the  accuracy  of  the  selected  features.The  genetic  algorithm  resultedin a  similar  number  of features to the  greedy  search.However,  the individual  elements  in  genetic  algorithm  are evaluated from high performance ‘offspring’ from their parents, as a result, this leads to an improvement of the dataset and their features to the given goal.

## 3) Training and Testing the Classifier
The unlabelled dataset was divided in a 70-30 split. A feedforward neural network was designed in MATLAB. The number of hidden layers and learning rate were iteratively optimized. Steps were taken to ensure the classifier does not overfit the training set. The number of hidden layers was kept low (between 1 and 3), to avoid over-estimating the complexity of the target problem.

![Alt text](neuralnet.png?raw=true "Title")


A number of training functions were then tested, and were assessed based on the ability of the network to generalize to the hold-out set.These include ‘trainlm’ which is based on Levenberg-Marquardt backpropagation, ‘trainscg’ which is scaled conjugate gradient backpropagation, and ‘trainbr’ which is a bayesian regularization backpropagation algorithm based on Levenberg-Marquardt optimization. ‘Trainbr’ seemed to perform the best, when used with its default training parameters.

## 4) Meta-Learning Approach
Given the success of undersampling in increasing the performance of the classifier, a variation on bagging was explored to further improve the sensitivity and recall.

![Alt text](process.png?raw=true "Title")

As the diagram suggests, at the testing phase, the networks are applied to the test set and voting by committee was performed to determine the final decision. The decision threshold of the vote was tweaked to find the optimum operating point along the P-R curve. The use of bagging improved the optimal operating point of the classifier slightly

## 5) Results
Using multilayer perceptron as a classifier, genetic algorithm as a feature selector, and voting-based query as an active learning method, the following precision-recall curve was generated.

![Alt text](Results/PRCurve.png?raw=true "Title")

After fine-tuning our decision threshold, we were able to achieve precision and recall rate similarly: 30.0% and 30.6%. To estimate classifier performance on the blind test set, we used bootstrapping, to generate 200 class imbalanced samples from the hold-out test set. The classifier’s precision at a recall rate of 30% was computed at each step.

![Alt text](Results/Hist.png?raw=true "Title")

