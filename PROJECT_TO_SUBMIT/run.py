import numpy as np
import matplotlib.pyplot as plt
import sys
from implementations import *

##############################################################################
######################## SET PREPROCESSING PARAMETERS ########################
##############################################################################

# Set if we want to replace invalid values (-999) by zero. 
ReplaceToZero_FLAG = True

# Set if we want to remove the highly correlated columns from the datasets.
RemoveHCColumns_FLAG = True
CorrelationThreshold = 0.8

##############################################################################
############################### PREPROCESSING  ###############################
##############################################################################
print('----------PREPROCESSING----------')

# Reload the data, the first part had explanatory purposes. Here we want to start from scratch, and to repeat step by step
# the manipulations that we applied 
print('Loading training data ...')
yb, input_data, ids = load_csv_data("../data/train.csv", sub_sample=False)
print('Train data have been loaded.')

# Replace -999 with zeros
if ReplaceToZero_FLAG:
    input_data = replaceWithZero(input_data,-999)
    print('Invalid values (i.e. -999) replaced with 0.')
    
# Now we are going to divide the dataset in 3 different datasets, according to the number of jets (feature 22 of dataset).
# Since in the new division some columns will have constant values, we will also delete those columns and save the indices in     # order do the same preprocessing when we want to use our model to predict.

# Finally, note that we keep the mean and the std of the train set. These parameters will be used to perform the standardization  # in the prediction phase (on the test set). 

print('Performing division in three sub-sets (depending on the number of jets).')

idxs = indices_jet_division(input_data)

x0, x1, x2 = data_split_with_jet_division(input_data,idxs)
y0, y1, y2 = data_split_with_jet_division(yb,idxs)

# Only the first 2 subsets have const columns.
x0, idx_constants_removed0 = removeConstantColumns(x0)
x1, idx_constants_removed1 = removeConstantColumns(x1)


x0, mean_train0, std_train0 = standardize ( x0 )
x1, mean_train1, std_train1 = standardize ( x1 )
x2, mean_train2, std_train2 = standardize ( x2 )

# We remove Highly Correlated (HC) columns.
if RemoveHCColumns_FLAG:
    x0, idx_HC_removed0 = removeHighCorrelatedColumns(x0, CorrelationThreshold)
    x1, idx_HC_removed1 = removeHighCorrelatedColumns(x1, CorrelationThreshold)
    x2, idx_HC_removed2 = removeHighCorrelatedColumns(x2, CorrelationThreshold)
    print('Highly correlated columns removed.')
else:
    idx_HC_removed0 = []
    idx_HC_removed0 = []
    idx_HC_removed0 = []


##############################################################################
################################# Training  ##################################
##############################################################################
print('----------TRAINING----------')

# Grid search to find the best degree and the best lambda.
# For each couple (lambda, degree) we trained the model using ridge regression with 4-fold cross validation. Since this is not a  # classifier, the performance of the model is evaluated using the mean of the 4 accuracies (i.e. the percentage of correct        # predictions) on the validations set of the cross validation.

degrees0 = np.arange(8,15)
lambdas0 = np.linspace(1e-7,0.0001,25) 

degrees1 = np.arange(8,15)
lambdas1 = np.linspace(1e-5,0.0001,25) 

degrees2 = np.arange(10,17)
lambdas2 = np.linspace(1e-5,0.01,25) 

best_lambda_acc0, best_degree_acc0, accuracy0 = grid_search_hyperparam_with_RidgeCV(y0, x0, lambdas0, degrees0)

best_lambda_acc1, best_degree_acc1, accuracy1 = grid_search_hyperparam_with_RidgeCV(y1, x1, lambdas1, degrees1)

best_lambda_acc2, best_degree_acc2, accuracy2 = grid_search_hyperparam_with_RidgeCV(y2, x2, lambdas2, degrees2)


print('Best hyperparameters:')
print(f'Model with 0 jets:  lambda = {best_lambda_acc0}, degree = {best_degree_acc0}, accuracy = {np.max(accuracy0)}')
print(f'Model with 1 jets:  lambda = {best_lambda_acc1}, degree = {best_degree_acc1}, accuracy = {np.max(accuracy1)}')
print(f'Model with >1 jets: lambda = {best_lambda_acc2}, degree = {best_degree_acc2}, accuracy = {np.max(accuracy2)}')


N0 = x0.shape[0]
N1 = x1.shape[0]
N2 = x2.shape[0]

# This is a good estimate of the total performance of our model
TOTAccuracy = ( N0*np.max(accuracy0) + N1*np.max(accuracy1) + N2*np.max(accuracy2) ) / ( N0 + N1 + N2 )
print(f'\n\n Our validation set reached a global accuracy of {TOTAccuracy}')

print('Retraining our model on the whole train (sub)sets using the obtained hyperparameters.')
# Now we retrain our model using the obtained hyperparameters and the full train (sub)sets.
x0_augmented = build_poly(x0, int(best_degree_acc0[0]))
best_w_acc0,_ = ridge_regression(y0,x0_augmented,best_lambda_acc0[0])

x1_augmented = build_poly(x1, int(best_degree_acc1[0]))
best_w_acc1,_ = ridge_regression(y1,x1_augmented,best_lambda_acc1[0])

x2_augmented = build_poly(x2, int(best_degree_acc2[0]))
best_w_acc2,_ = ridge_regression(y2,x2_augmented,best_lambda_acc2[0])

# Visualization of the best (degree,lambda) combination in the case of the subset with 1 jet
plot_grid_search(lambdas1, degrees1, accuracy1, 'grid_search.eps')

##############################################################################
################################ Prediction  #################################
##############################################################################
print('----------PREDICTION----------')

print('Loading test data...')
_, test_data, ids_test = load_csv_data("../data/test.csv", sub_sample=False)
num_tests = test_data.shape[0]
print('Test data have been loaded.')

print('Repeating the preprocessing steps on the test set')

# Replacing invalid values to 0.
if ReplaceToZero_FLAG:
    test_data = replaceWithZero(test_data,-999)
    print('Invalid values (-999) replaced with 0.')

# Jets division.
idxsTest = indices_jet_division(test_data)
x0Test, x1Test, x2Test = data_split_with_jet_division(test_data,idxsTest)
print('Division in subset performed.')

# Remove constant columns.
x0Test = np.delete(x0Test, idx_constants_removed0, axis=1)
x1Test = np.delete(x1Test, idx_constants_removed1, axis=1)
print('Constant columns removed.')


# Standardization with train parameters.
x0Test,_,_ = standardize ( x0Test, mean_train0, std_train0 )
x1Test,_,_ = standardize ( x1Test, mean_train1, std_train1 )
x2Test,_,_ = standardize ( x2Test, mean_train2, std_train2 )
print('Standardization performed.')

# Removing higly correlated columns.
if(RemoveHCColumns_FLAG):
    for i in idx_HC_removed0:
        x0Test = np.delete(x0Test,i,axis=1)
    for i in idx_HC_removed1:
        x1Test = np.delete(x1Test,i,axis=1)
    for i in idx_HC_removed2:
        x2Test = np.delete(x2Test,i,axis=1)
    print('Highly correlated columns removed.')


# Feature expansion according to the best degree found in the train set.
x0Test = build_poly(x0Test, int(best_degree_acc0[0]))
x1Test = build_poly(x1Test, int(best_degree_acc1[0]))
x2Test = build_poly(x2Test, int(best_degree_acc2[0]))
print('Feature expansion of test (sub)sets performed.')

# For each subset, predict its final label.
y_pred0 = predict_labels(best_w_acc0,x0Test)
y_pred1 = predict_labels(best_w_acc1,x1Test)
y_pred2 = predict_labels(best_w_acc2,x2Test)

# Reassemply the whole prediction.
y_pred = np.ones(num_tests)
y_pred[idxsTest[0]] = y_pred0
y_pred[idxsTest[1]] = y_pred1
y_pred[idxsTest[2]] = y_pred2
print('Final prediction performed.')


create_csv_submission(ids_test, y_pred, 'FinalModel_WithoutHCColumns.csv')
print('The .csv file containing our final prediction has been created.')
print('You can find it in the folder containing the run.py file!')
print('Group members: Nicola Ischia, Marion Perier, Leonardo Pollina.')


