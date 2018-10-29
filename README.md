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
(e.g. small norm of the gradient), thus there is no threshold parameter asked in the functions. 
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
Notice that this preprocessing step of feature selection was not used because final results were not satisfactory. 
</ul>

## proj1_helpers.py
Here there are the three functions provided by the professors:
  <ol>
  <li> <b> load_csv_data() </b> </li>
To load the train and test data with the corresponding labels (when provided) from a .csv file.
  <li> <b> predict_labels() </b> </li>
Function to predict labels in the case of normal regression methods (not logistic). <br>
It uses labels {-1,1}.
  <li> <b> create_csv_submission() </b> </li>
Function to create the .csv file containing the final labels to submit. 
</ol>

## run.py
This Python file contains the code used to implement all the different steps of our best (final) model, that is Ridge Regression. <br>
The code is divided in subsections: 
  <ul>
  <li> Importation of important libraries and other .py files. This includes the numpy and matplotlib.pyplot libraries and      the file implementations.py. This gives access to all the necessary functions. </li>
  <li> <b> Preprocessing </b> </li>
      <ol>   
      <li> Preprocessing parameters are set. Please note that if ReplaceToZero_FLAG is set to false, the convertion of invalid values to zero will not be performed. <br>
With the same logic, if RemoveHCColumns_FLAG is set to false, highly correlated (HC) columns will be kept in the datasets.<br> CorrelationThreshold indicates the threshold used to decide whether two features are HC or not. <br>
    For our best model, both the flag parameters are set to <b> True </b> , and the threshold is <b> 0.8 </b>. </li>
      <li> Loading of the training data using the provided function load_csv_data(). We obtain the train data and the corresponding labels. </li>
      <li> Invalid values (<b> -999 </b>) are replaced by 0.</li>
      <li> Division of the data set in the three subsets depending on the number of jets.</li>
      <li> Constant columns are removed from the three datasets. These columns are not supposed to give any useful information. The indeces of such columns are stocked in order to remove the same colums in the test set. </li>
      <li> The three sets are standardized. Notice that mean and std of each set are stocked in order to use these parameters for the preprocessing of the test set. </li>
      <li> Highly correlated columns are removed from each subset. Notice that indices of such columns are stocked in order to use these parameters for the preprocessing of the test set.</li>
  </ol>
  <li> <b> Training </b> </li>
      <ol>
        <li> Estimation of best hyperparameters (<b>degree</b> for feature expansion, and <b>lambda</b> for the regularization term)for Ridge Regression. The optimization is performed using cross-validation. <br>
        The best combination of hyperparameters was chosen depending of the maximum accuracy obtained.</li>
        <li> Training of the model on the whole data set (still divided in three subsets) using the hyperparameters previously found. <br> In this way, the weights are obtained. </li>
  </ol>
  <li> <b> Submission </b> </li>
       <ol>
       <li> Loading of the test data.</li>
       <li> Division in three subsets depending on the number of jets.</li>
       <li> Deletion of constant columns using the indices stocked from the train set. </li>
       <li> Standardization using for each subset the mean and std of the corresponding train set. </li>
       <li> Deletion of HC columns using the indices stocked from the train set. </li>
       <li> Features expansion using the optimal degree previously found.</li>
       <li> Prediction of final labels using the provided function predict_labels() </li>
       <li> Reassembly of the three different prediction vectors into a unique vector. </li>
       <li> Creation of the .csv file using the provided function create_csv_submission() </li>
  </ol>
</ul>

## General information
<ul>
  <li> Notice that the submission file will be saved in your current folder (the folder containing the run.py file). </li>
  <li> In order to run the file run.py, please use the terminal and type the command "python run.py" </li>
</ul>


## Software versions
Please, notice that all the code included in this project was written using Python Version 3.6 and only NumPy (version 1.14.3) and Matplolib (for visualization tasks) libraries were employed.

<br>
<p align="right"/>
Group members: <i> Nicola Ischia, Marion Perier, Leonardo Pollina. <i>
