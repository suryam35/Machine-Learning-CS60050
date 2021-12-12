import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time 


def import_data(filename):
	# Imports the data from the given ‘diabetes.csv’ file and converts it to a numpy array 
	# with the elements being in float data type.

	train_XY = pd.read_csv(filename)
	# print(type(train_XY))
	return train_XY

def label_encoding(train_XY_data):
	# This function does the encoding of the categorical attributes in the format {a:1, b:2, …} 
	# for each column which contains categorical values.

	header=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
	for h in header:
		new_header = h + "_"
		train_XY_data[h] = train_XY_data[h].astype('category')
		train_XY_data[new_header] = train_XY_data[h].cat.codes
		train_XY_data = train_XY_data.drop(h, axis = 1)
	return train_XY_data

def one_hot_encoder(train_XY_data):
	# This function does one hot encoding of the columns which contain categorical attributes.

	header=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
	for h in header:
		one_hot = pd.get_dummies(train_XY_data[h], prefix = h)
		train_XY_data = train_XY_data.drop(h, axis = 1)
		train_XY_data = train_XY_data.join(one_hot)
	return train_XY_data

def standardize(train_X_data):
	# This function is used to standardize the column values using the formula X’ =  (X - µ) / σ

	X = StandardScaler().fit_transform(train_X_data)
	return X

def get_train_test_validation_split(train_XY_data):
	# This function randomly shuffles the data and does a 70:10:20 split to get the training, validation
	# and testing data. The data is further divided into train_X, train_Y, test_X, test_Y, validate_X, validate_Y 
	# sets which denote respectively the training attributes, target values for training, testing attributes and 
	# target values for testing, validation attributes and target values for validation.

	train_XY_data = train_XY_data.sample(frac = 1)
	all_X = train_XY_data.iloc[:, :-1]
	all_Y = train_XY_data.iloc[:, -1]
	all_X.to_numpy()
	all_Y.to_numpy()
	all_X = standardize(all_X)
	size = all_X.shape[0]
	train_X, validate_X, test_X = all_X[0: int(0.7*size)], all_X[int(0.7*size):int(0.8*size)], all_X[int(0.8*size):]
	train_Y, validate_Y, test_Y = all_Y[0: int(0.7*size)], all_Y[int(0.7*size):int(0.8*size)], all_Y[int(0.8*size):]
	return train_X, train_Y, test_X, test_Y, validate_X, validate_Y


def get_plot_pca(finalDf, file_name, i):
	# This function is used to plot the data using the 2 principal components on a 2-D graph in which the data points 
	# of the same class have the same color.

	fig = plt.figure(i, figsize = (20,20))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_title('2 component PCA', fontsize = 20)

	targets = [0, 1]
	colors = ['r', 'g']
	for target, color in zip(targets,colors):
	    indicesToKeep = finalDf['Outcome'] == target
	    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
	ax.legend(targets)
	ax.grid()
	plt.savefig(file_name)

def get_plot_lda(finalDf, file_name, i):
	# This function is used to plot the data using the 1 component obtained after doing LDA on a 2-D graph in which data points 
	# of the same class have the same color and the y-coordinate of each point is assumed to be 0.

	fig = plt.figure(i, figsize = (20,20))
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel('LDA Component 1', fontsize = 15)
	ax.set_title('1 component LDA', fontsize = 20)
	targets = [0, 1]
	colors = ['r', 'g']
	for target, color in zip(targets,colors):
	    indicesToKeep = finalDf['Outcome'] == target
	    ax.scatter(finalDf.loc[indicesToKeep, 'LDA axis - 1'], [0 for i in finalDf.loc[indicesToKeep, 'LDA axis - 1']], c = color, s = 50)
	ax.legend(targets)
	ax.grid()
	plt.savefig(file_name)
	

def get_2pca_plot(principalComponents, train_Y):
	# This function calls the get_plot_pca() function with the required parameters to get the plot.

	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
	finalDf = pd.concat([principalDf, train_Y], axis = 1)
	get_plot_pca(finalDf, '2_component_PCA.jpeg', 1)

def get_2lda_plot(ld, train_Y):
	# This function calls the get_plot_lda() function with the required parameters to get the plot.

	principalDf = pd.DataFrame(data = ld, columns = ['LDA axis - 1'])
	finalDf = pd.concat([principalDf, train_Y], axis = 1)
	get_plot_lda(finalDf, '1_component_LDA.jpeg', 2)


def do_pca(train_X):
	# This function is used to do the principal component analysis with 2 components and returns the 
	# transformed training data and the projection matrix.

	pca = PCA(n_components=2)
	pca.fit(train_X)
	principalComponents = pca.transform(train_X)
	return principalComponents, pca

def get_accuracy(pred_Y, target_Y):
	# This function is used to get the accuracy score of the predicted values against the target values.

	accuracy = accuracy_score(pred_Y, target_Y)
	return accuracy*100


def do_svm(train_X, train_Y, validate_X, validate_Y, pc_model):
	# This function is used to implement support vector machines for different kernels with varying hyperparameters 
	# and returns the best kernel and hyperparameter for which the validation accuracy is the maximum.

	best_kernel = ""
	best_degree = -1
	best_coef = -1
	best_ceof_sigmoid = -1
	best_accuracy = 0
	kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
	validate_pc = pc_model.transform(validate_X)
	for kernel_ in kernel_list:
		if kernel_ == 'poly':
			for d in range(1, 5):
				for c in range(1, 5):
					clf = SVC(kernel = kernel_, degree = d, coef0 = c)
					clf.fit(train_X, train_Y)
					pred_Y = clf.predict(validate_pc)
					accuracy = get_accuracy(pred_Y, validate_Y)
					print("Kernel = ", kernel_, ", Degree = " , d, ", coef0 = " , c,", Validation Accuracy = " , accuracy)
					if accuracy > best_accuracy:
						best_accuracy = accuracy
						best_degree = d
						best_kernel = kernel_
						best_coef = c
		elif kernel_ == 'sigmoid':
			for c in range(1, 5):
				clf = SVC(kernel = kernel_, coef0 = c)
				clf.fit(train_X, train_Y)
				pred_Y = clf.predict(validate_pc)
				accuracy = get_accuracy(pred_Y, validate_Y)
				print("Kernel = ", kernel_, ", coef0 = ",c, ", Validation Accuracy = " , accuracy)
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					best_kernel = kernel_
					best_degree = -1
					best_ceof_sigmoid = c
		else:
			clf = SVC(kernel = kernel_)
			clf.fit(train_X, train_Y)
			pred_Y = clf.predict(validate_pc)
			accuracy = get_accuracy(pred_Y, validate_Y)
			print("Kernel = ", kernel_, ", Validation Accuracy = " , accuracy)
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_kernel = kernel_
				best_degree = -1
				best_ceof_sigmoid = -1
				best_coef = -1
	return best_kernel, best_degree, best_accuracy, best_coef, best_ceof_sigmoid

def get_test_accuracy(best_kernel, best_degree, test_X, test_Y, pc, train_Y, pc_model, best_coef, best_ceof_sigmoid):
	# This function is used to get the accuracy on the test data by implementing a svm classifier on the 
	# best kernel and hyper parameters obtained.

	clf = None
	if best_degree != -1:
		clf = SVC(kernel = best_kernel, degree = best_degree, coef0 = best_coef)
	elif best_kernel == 'sigmoid':
		clf = SVC(kernel = best_kernel, coef0 = best_ceof_sigmoid)
	else:
		clf = SVC(kernel = best_kernel)
	clf.fit(pc, train_Y)
	test_pc = pc_model.transform(test_X)
	pred_Y = clf.predict(test_pc)
	test_accuracy = get_accuracy(pred_Y, test_Y)
	return test_accuracy

def do_LDA(train_X, train_Y):
	# This function is used to perform Linear Discriminant Analysis of the data and returns the transformed 
	# training data and the projection matrix.

	lda = LDA(n_components = 1)
	lda.fit(train_X, train_Y)
	ldacomponents = lda.transform(train_X)
	return ldacomponents, lda

def step_3_for_pca(pc, train_Y, validate_X, validate_Y, pc_model, test_X, test_Y):
	# This function calls the do_svm() function with the data obtained after 2 component PCA and gets the best 
	# kernel and hyperparameters and then calls the get_test_accuracy() function to report the accuracy on the test data.

	print("************** PCA *************")
	best_kernel, best_degree, best_accuracy, best_coef, best_ceof_sigmoid = do_svm(pc, train_Y, validate_X, validate_Y, pc_model)
	print("Best Accuracy = ", best_accuracy, ", Best kernel = " , best_kernel, ", Best Degree (-1 if not poly) = ", best_degree)
	test_accuracy = get_test_accuracy(best_kernel, best_degree, test_X, test_Y, pc, train_Y, pc_model, best_coef, best_ceof_sigmoid)
	print("Test accuracy = " , test_accuracy)

def step_3_for_lda(ld, train_Y, validate_X, validate_Y, lda_model, pc, pc_model, test_X, test_Y):
	# This function calls the do_svm() function with the data obtained after 1 component LDA and gets the best kernel and 
	# hyperparameters and then calls the get_test_accuracy() function to report the accuracy on the test data.
	
	print("\n\n************** LDA *************")
	# validate_X = pc_model.transform(validate_X)
	best_kernel, best_degree, best_accuracy, best_coef, best_ceof_sigmoid = do_svm(ld, train_Y, validate_X, validate_Y, lda_model)
	print("Best Accuracy = ", best_accuracy, ", Best kernel = " , best_kernel, ", Best Degree (-1 if not poly) = ", best_degree)
	# test_X = pc_model.transform(test_X)
	test_accuracy = get_test_accuracy(best_kernel, best_degree, test_X, test_Y, ld, train_Y, lda_model, best_coef, best_ceof_sigmoid)
	print("Test accuracy = " , test_accuracy)
	


if __name__ == '__main__':
	# Question 1
	train_XY_data = import_data('diabetes.csv')
	# train_XY_data = one_hot_encoder(train_XY_data)
	# train_XY_data = label_encoding(train_XY_data)
	train_X, train_Y, test_X, test_Y, validate_X, validate_Y = get_train_test_validation_split(train_XY_data)

	# Question 2
	pc, pc_model = do_pca(train_X)
	get_2pca_plot(pc, train_Y)

	# Question 3
	step_3_for_pca(pc, train_Y, validate_X, validate_Y, pc_model, test_X, test_Y)

	# Question 4
	ld, lda_model = do_LDA(train_X, train_Y)
	get_2lda_plot(ld, train_Y)

	# question 5
	step_3_for_lda(ld, train_Y, validate_X, validate_Y, lda_model, pc, pc_model, test_X, test_Y)


# ************** Example output of the code **************


# ************** PCA **************
# Kernel =  linear , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  1 , coef0 =  1 , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  1 , coef0 =  2 , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  1 , coef0 =  3 , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  1 , coef0 =  4 , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  2 , coef0 =  1 , Validation Accuracy =  81.81818181818183
# Kernel =  poly , Degree =  2 , coef0 =  2 , Validation Accuracy =  81.81818181818183
# Kernel =  poly , Degree =  2 , coef0 =  3 , Validation Accuracy =  81.81818181818183
# Kernel =  poly , Degree =  2 , coef0 =  4 , Validation Accuracy =  81.81818181818183
# Kernel =  poly , Degree =  3 , coef0 =  1 , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  3 , coef0 =  2 , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  3 , coef0 =  3 , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  3 , coef0 =  4 , Validation Accuracy =  77.92207792207793
# Kernel =  poly , Degree =  4 , coef0 =  1 , Validation Accuracy =  81.81818181818183
# Kernel =  poly , Degree =  4 , coef0 =  2 , Validation Accuracy =  81.81818181818183
# Kernel =  poly , Degree =  4 , coef0 =  3 , Validation Accuracy =  81.81818181818183
# Kernel =  poly , Degree =  4 , coef0 =  4 , Validation Accuracy =  81.81818181818183
# Kernel =  rbf , Validation Accuracy =  80.51948051948052
# Kernel =  sigmoid , coef0 =  1 , Validation Accuracy =  70.12987012987013
# Kernel =  sigmoid , coef0 =  2 , Validation Accuracy =  68.83116883116884
# Kernel =  sigmoid , coef0 =  3 , Validation Accuracy =  67.53246753246754
# Kernel =  sigmoid , coef0 =  4 , Validation Accuracy =  76.62337662337663
# Best Accuracy =  81.81818181818183 , Best kernel =  poly , Best Degree (-1 if not poly) =  2 , Best Coefficient (-1 if not poly) =  1 , Best Sigmoid Coefficient (-1 if not sigmoid) =  -1
# Test accuracy =  72.72727272727273


# ************** LDA *************
# Kernel =  linear , Validation Accuracy =  76.62337662337663
# Kernel =  poly , Degree =  1 , coef0 =  1 , Validation Accuracy =  76.62337662337663
# Kernel =  poly , Degree =  1 , coef0 =  2 , Validation Accuracy =  76.62337662337663
# Kernel =  poly , Degree =  1 , coef0 =  3 , Validation Accuracy =  76.62337662337663
# Kernel =  poly , Degree =  1 , coef0 =  4 , Validation Accuracy =  76.62337662337663
# Kernel =  poly , Degree =  2 , coef0 =  1 , Validation Accuracy =  80.51948051948052
# Kernel =  poly , Degree =  2 , coef0 =  2 , Validation Accuracy =  80.51948051948052
# Kernel =  poly , Degree =  2 , coef0 =  3 , Validation Accuracy =  80.51948051948052
# Kernel =  poly , Degree =  2 , coef0 =  4 , Validation Accuracy =  80.51948051948052
# Kernel =  poly , Degree =  3 , coef0 =  1 , Validation Accuracy =  80.51948051948052
# Kernel =  poly , Degree =  3 , coef0 =  2 , Validation Accuracy =  80.51948051948052
# Kernel =  poly , Degree =  3 , coef0 =  3 , Validation Accuracy =  80.51948051948052
# Kernel =  poly , Degree =  3 , coef0 =  4 , Validation Accuracy =  80.51948051948052
# Kernel =  poly , Degree =  4 , coef0 =  1 , Validation Accuracy =  79.22077922077922
# Kernel =  poly , Degree =  4 , coef0 =  2 , Validation Accuracy =  79.22077922077922
# Kernel =  poly , Degree =  4 , coef0 =  3 , Validation Accuracy =  79.22077922077922
# Kernel =  poly , Degree =  4 , coef0 =  4 , Validation Accuracy =  79.22077922077922
# Kernel =  rbf , Validation Accuracy =  79.22077922077922
# Kernel =  sigmoid , coef0 =  1 , Validation Accuracy =  68.83116883116884
# Kernel =  sigmoid , coef0 =  2 , Validation Accuracy =  61.038961038961034
# Kernel =  sigmoid , coef0 =  3 , Validation Accuracy =  59.74025974025974
# Kernel =  sigmoid , coef0 =  4 , Validation Accuracy =  68.83116883116884
# Best Accuracy =  80.51948051948052 , Best kernel =  poly , Best Degree (-1 if not poly) =  2 , Best Coefficient (-1 if not poly) =  1 , Best Sigmoid Coefficient (-1 if not sigmoid) =  -1
# Test accuracy =  72.07792207792207