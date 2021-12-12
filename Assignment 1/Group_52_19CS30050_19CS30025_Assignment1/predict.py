import numpy as np
import math

def predict_target_values(root, test_X):
	'''
	This function calls the predict() function for predicting 
	the target values of each and every data present in the test set.

	Parameters
	----------
	root:   root node of the decision tree with which the target value is predicted
	test_X: test set for which we want the predicted values.    
   
	Returns
	-------
	predicted_class: the predicted class for the dataset.
	'''
	pred = []
	for i in range(len(test_X)):
		a = test_X[i]
		prediction = predict(root, a)
		pred.append(prediction)
	pred = np.array(pred)
	return pred

def check_accuracy(root, test_X, test_Y):
	'''
	This function calculates the accuracy given the root, data attributes and 
	target values

	Parameters
	----------
	root:   root node of the decision tree for which the accuracy is calculated
	test_X: test set for which we want to calculate the accuracy
	test_Y:	target values for the test set
   
	Returns
	-------
	predicted_class: the predicted class for the dataset.
	'''
	pred_Y = predict_target_values(root, test_X)
	# print(test_Y.shape , pred_Y.shape)
	matching_count = 0
	for i in range(len(test_Y)):
		if test_Y[i] == pred_Y[i]:
			matching_count += 1
	accuracy = matching_count/len(test_Y)
	return accuracy


def predict(root, X):
	'''
	This function is used to predict the value of one single 
	data set. It follows the path in the tree based on the 
	attribute used at that node for splitting.

	Parameters
	----------
	root:   root node of the decision tree with which the target value is predicted
	X:      single data set provided
   
	Returns
	-------
	predicted_class: the predicted class for the dataset.
	'''
	feature_index = root.feature_index
	threshold = root.threshold
	value = X[feature_index]
	if value < threshold:
		if root.left is None:
			return root.predicted_class
		else:
			return predict(root.left, X)
	else:
		if root.right is None:
			return root.predicted_class
		else:
			return predict(root.right, X)
