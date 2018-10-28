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
The functions are divided in different groups, which are explained below.
  <ul>
  <li> <b> Requested methods </b> </li>
The six functions asked in report guidelines (least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression).
All these functions return (w, loss). Notice that, in the case of iterative functions, we decided to not add a stopping criteria
(e.g. small norm of the gradient), that is there is no threshold parameter asked in the functions. 
  <li> <b> Preprocessing </b> </li>
All the functions needed by our final regression model for the preliminary data analisys and the preprocessing.
  <li> run.py </li> 
</ul>
