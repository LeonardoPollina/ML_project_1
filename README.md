# Machine learning project 1 (CS-433)
This folder contains the files used to solve the Higgs Boson Machine Learning Challenge, using the ATLAS dataset.

In this folder there are 3 files:
  <ul>
  <li> implementations.py </li>
  <li> proj1_helpers.py </li>
  <li> run.py </li> 
</ul>

## implementations.py
Here there are all the implemented functions used for the different steps of our model derivation.
Functions are divided in different groups, which are explained below.
  <ul>
  <li> <b> Requested methods </b> </li>
It contains the six functions asked in report guidelines (least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression).
All these functions return (w, loss). <br>
Notice that, in the case of iterative functions, we did not add a stopping criteria
(e.g. small norm of the gradient), that is there is no threshold parameter asked in the functions. 
  <li> <b> Preprocessing </b> </li>
All the functions needed for the preliminary data analysis and preprocessing of our final regression model. 
Functions are in the same order as they are called in the run.py file.
  <li> <b> Loss functions </b> </li> 
All the functions required to compute the loss of different models. <br>
  In particular, MSE, MAE, RMSE, logistic and regularized logistic losses are implemented. 
  <li> <b> Gradient functions </b> </li> 
All the functions required to compute the gradient of different loss functions. In particular, MSE, MAE, logistic and regularized logistic gradients are implemented.  
  <li> <b> Predictions </b> </li> 
Functions for labels prediciton for logistic regression (classification model). In this case the labels are {0,1} and thus, the threshold to perform classification is set to 0.5 . <br>
Notice that functions for labels prediction for regression models are contained in 'proj1_helpers.py'.
  <li> <b> Newton functions </b> </li> 
All the functions needed to apply Netwon method for classification. 
  <li> <b> Useful functions </b> </li> 
All the additional functions which can be used to manipulate data, such as subsampling of data, polynomial expansion, counting and replacing of a chosen invalid value, applying the sigmoid function etc.
  <li> <b> PCA </b> </li>
Functions needed to compute PCA. <br>
Notice that this preprocessing step for feature selection was not used because final results were not satisfactory. 
</ul>

## proj1_helpers.py
Here there are the three functions provided by the professors. 
  <ol>
  <li> load_csv_data() </li>
To load the train and test data with the corresponding labels (when provided) from a .csv file.
  <li> predict_labels() </li>
Function to predict labels in the case of normal regression methods (not logistic). <br>
It uses labels {-1,1}.
  <li> create_csv_submission() </li>
Function to create the .csv file containing the final labels to submit. 
</ol>
