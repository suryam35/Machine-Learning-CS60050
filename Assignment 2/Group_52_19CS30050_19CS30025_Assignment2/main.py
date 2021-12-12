import numpy as np
import math
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

def convert(x):
	return float(x == b'Male')

def import_data(file_name):
	train_XY = np.genfromtxt(file_name, delimiter=',', dtype=np.float64, skip_header=1, converters = {1: convert})
	# print(type(train_XY))
	return train_XY


def fill_missing_values(train_XY_data):
	column_sum = 0
	column_length = 0
	for i in range(train_XY_data.shape[0]):
		if not np.isnan(train_XY_data[i][9]):
			column_sum += train_XY_data[i][9]
			column_length += 1

	for i in range(train_XY_data.shape[0]):
		if np.isnan(train_XY_data[i][9]):
			train_XY_data[i][9] = column_sum/column_length

	return train_XY_data


def normalize_data(train_XY_data):
	train_XY_data = fill_missing_values(train_XY_data)
	np.random.shuffle(train_XY_data)
	train_X = train_XY_data[:, :-1]
	train_Y = train_XY_data[:, -1]
	for j in range(train_X.shape[1]):
		column_min = np.min(train_X[: , j])
		column_max = np.max(train_X[: , j])
		train_X[:, j] = (train_X[: , j] - column_min)/(column_max - column_min)

	size = len(train_XY_data)
	return train_X[:int(0.8*size)], train_Y[:int(0.8*size)], train_X[int(0.8*size):] , train_Y[int(0.8*size):]


def get_validation_and_train_data(train_XY_data, seed):
	np.random.seed(seed)
	np.random.shuffle(train_XY_data)
	size = len(train_XY_data)
	train_XY = train_XY_data[:int(0.8*size)]
	test_XY = train_XY_data[int(0.8*size):]
	train_X = train_XY[: , :-1]
	train_Y = train_XY[:, -1]
	test_X = test_XY[: , :-1]
	test_Y = test_XY[:, -1]
	return train_X, train_Y, test_X, test_Y

def get_euclidean_distance(X1, X2):
	distance = 0
	for i in range(len(X1)):
		distance += (X1[i] - X2[i])**2
	return distance**0.5


def get_min_distance_index(centroids, row):
	distance = []
	for centroid in centroids:
		e_d = get_euclidean_distance(centroid, row)
		distance.append(e_d)
	return distance.index(min(distance))

def update_centroids(centroids, centroid_dictionary):
	new_centroids = []
	for key, value in centroid_dictionary.items():
		temp_row = [0 for i in range(len(value[0]))]
		for row in value:
			for i in range(len(row)):
				temp_row[i] += row[i]
		for i in range(len(row)):
			temp_row[i] /= len(value)
		new_centroids.append(temp_row)
	return new_centroids


def random_centroids(K, train_X):
	initial_random_centroids = []

	while len(initial_random_centroids) != K:
		index = np.random.randint(0, train_X.shape[0])
		if index not in initial_random_centroids:
			initial_random_centroids.append(index)
	return initial_random_centroids


def Kmeans_clustering(train_X, train_Y, K, max_iterations, initial_random_centroids):

	centroids = [train_X[i] for i in initial_random_centroids]
	for iteration in range(max_iterations):
		centroid_dictionary = defaultdict(lambda: [])
		centroid_dictionary_class_label = defaultdict(lambda: [0,0])
		for i in range(train_X.shape[0]):
			index_of_min = get_min_distance_index(centroids, train_X[i])
			centroid_dictionary[index_of_min].append(train_X[i])
			centroid_dictionary_class_label[index_of_min][int(train_Y[i])-1] += 1
		centroids = update_centroids(centroids, centroid_dictionary)

	# get_class_labels
	class_labels_of_centroids = []
	for key, value in centroid_dictionary_class_label.items():
		if(value[0] > value[1]):
			class_labels_of_centroids.append(1)
		else:
			class_labels_of_centroids.append(2)

	return centroids, class_labels_of_centroids

def predict(train_X, centroids, class_labels):
	pred_Y = []
	for row in train_X:
		index = get_min_distance_index(centroids, row)
		pred_Y.append(class_labels[index])
	return pred_Y

def performance_using_ground_truth(centroids, class_labels, train_Y, train_X):
	pred_Y = predict(train_X, centroids, class_labels)

	total = 0
	for i in range(len(pred_Y)):
		total += int(pred_Y[i] == train_Y[i])

	total /= len(pred_Y)

	print("Ground truth: ", total*100)
	return total*100


def performance_using_homogenity(centroids, class_labels, train_Y, train_X):
	pred_Y = predict(train_X, centroids, class_labels)
	h_score = homogeneity_score(train_Y, pred_Y)
	print("Homogenity: ", h_score)
	return h_score


def performance_using_ARI(centroids, class_labels, train_Y, train_X):
	pred_Y = predict(train_X, centroids, class_labels)
	ARI_score = adjusted_rand_score(train_Y, pred_Y)
	print("ARI: ", ARI_score)
	return ARI_score


def performance_using_NMI(centroids, class_labels, train_Y, train_X):
	pred_Y = predict(train_X, centroids, class_labels)
	NMI_score = normalized_mutual_info_score(train_Y, pred_Y)
	print("NMI: ", NMI_score)
	return NMI_score

def performance_using_silhouette(centroids, class_labels, train_Y, train_X):
	pred_Y = predict(train_X, centroids, class_labels)
	if(len(set(pred_Y)) == 1):
		s_score = 0
	else:
		s_score = silhouette_score(train_X, pred_Y)
	print("Silhouette: ", s_score)
	return s_score

def performance_using_calinski_harabasz(centroids, class_labels, train_Y, train_X):
	pred_Y = predict(train_X, centroids, class_labels)
	if(len(set(pred_Y)) == 1):
		c_score = 0
	else:
		c_score = calinski_harabasz_score(train_X, pred_Y)
	print("Calinski Harabasz score: ", c_score)
	return c_score


def get_plot(Y, X, file_name, i, xlabel, ylabel, name):
	plot1 = plt.figure(i)
	plt.plot(X, Y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(name)
	plt.savefig(file_name)


def permute_data(train_XY_data):
	train_XY_data = fill_missing_values(train_XY_data)
	np.random.shuffle(train_XY_data)
	train_X = train_XY_data[:, :-1]
	train_Y = train_XY_data[:, -1]
	for j in range(train_X.shape[1]):
		column_min = np.min(train_X[: , j])
		column_max = np.max(train_X[: , j])
		train_X[:, j] = (train_X[: , j] - column_min)/(column_max - column_min)

	size = len(train_XY_data)
	return train_X[:int(0.4*size)], train_X[int(0.4*size):int(0.8*size)], train_Y[:int(0.4*size)], train_Y[int(0.4*size):int(0.8*size)], train_X[int(0.8*size):], train_Y[int(0.8*size):]


def wangs_method(train_XY_data, c):
	average_accuracy = []
	for k in range(1, 50):
		accuracy_for_this_k = 0
		for i in range(c):
			train_X_1, train_X_2, train_Y_1, train_Y_2, test_X, test_Y = permute_data(train_XY_data)
			initial_random_centroids_1 = random_centroids(k, train_X_1)
			centroids_1, class_labels_1 = Kmeans_clustering(train_X_1, train_Y_1, k, 10, initial_random_centroids_1)
			initial_random_centroids_2 = random_centroids(k, train_X_2)
			centroids_2, class_labels_2 = Kmeans_clustering(train_X_2, train_Y_2, k, 10, initial_random_centroids_2)
			accuracy_1 = performance_using_ground_truth(centroids_1, class_labels_1, test_Y, test_X)
			accuracy_2 = performance_using_ground_truth(centroids_2, class_labels_2, test_Y, test_X)
			accuracy_for_this_k += accuracy_1 + accuracy_2
		accuracy_for_this_k /= (2*c)
		average_accuracy.append(accuracy_for_this_k)

	return average_accuracy.index(max(average_accuracy)) + 1, average_accuracy

def test_A_kmeans(K, train_XY_data):
	average_metric = []
	iteration_list = [i for i in range(1, 51)]
	for i in range(50):
		metric_score = 0
		train_X, train_Y, test_X, test_Y = normalize_data(train_XY_data)
		initial_random_centroids = random_centroids(K, train_X)
		for j in range(50):
			train_X, train_Y, test_X, test_Y = normalize_data(train_XY_data)
			centroids, class_labels = Kmeans_clustering(train_X, train_Y, K, 5, initial_random_centroids)
			metric_score += performance_using_NMI(centroids, class_labels, test_Y, test_X)
		metric_score /= 50
		average_metric.append(metric_score)

	get_plot(average_metric, iteration_list, 'test_A_kmeans.jpeg', 8, 'iteration count', 'average metric', 'average_metric VS iteration (fixed K)')
	x = statistics.mean(average_metric)
	print("Mean for k is :", x)
	x = statistics.stdev(average_metric)
	print("Standard Deviation for k is :", x)
	pass

def test_A_kplusplus(K, train_XY_data):
	average_metric = []
	iteration_list = [i for i in range(1, 51)]
	for i in range(50):
		metric_score = 0
		train_X, train_Y, test_X, test_Y = normalize_data(train_XY_data)
		initial_random_centroids = Kmeans_plus_plus(train_X, K)
		for j in range(50):
			train_X, train_Y, test_X, test_Y = normalize_data(train_XY_data)
			centroids, class_labels = Kmeans_clustering(train_X, train_Y, K, 5, initial_random_centroids)
			metric_score += performance_using_NMI(centroids, class_labels, test_Y, test_X)
		metric_score /= 50
		average_metric.append(metric_score)

	get_plot(average_metric, iteration_list, 'test_A_kplusplus.jpeg', 9, 'iteration count', 'average metric', 'average_metric VS iteration (fixed K)')
	x = statistics.mean(average_metric)
	print("Mean for k++ is :", x)
	x = statistics.stdev(average_metric)
	print("Standard Deviation for k++ is :", x)
	pass


def Kmeans_plus_plus(train_X, K):
	initial_random_centroids = []
	initial_random_centroids.append(np.random.randint(0, train_X.shape[0]))

	while len(initial_random_centroids) != K:
		farthest = -10000
		farthest_index = -1
		for i in range(train_X.shape[0]):
			row = train_X[i]
			if i in initial_random_centroids:
				continue
			best_distance = 100000
			for j in range(len(initial_random_centroids)):
				distance =get_euclidean_distance(row, train_X[initial_random_centroids[j]])
				if distance < best_distance:
					best_distance = distance
			if best_distance > farthest:
				farthest = best_distance
				farthest_index = i

		initial_random_centroids.append(farthest_index)
	return initial_random_centroids


def get_plot_for_indexes(train_XY_data):
	train_X, train_Y, test_X, test_Y = normalize_data(train_XY_data)
	K_list = [i for i in range(1, 50)]
	
	ground_truth_list = []
	ARI_list = []
	NMI_list = []
	homogeneity_list = []
	silhouette_list = []
	calinski_list = []
	for K_value in range(1, 50):
		initial_random_centroids = random_centroids(K_value, train_X)
		centroids, class_labels = Kmeans_clustering(train_X, train_Y, K_value, 100, initial_random_centroids)
		silhouette_list.append(performance_using_silhouette(centroids, class_labels, test_Y, test_X))
		ground_truth_list.append(performance_using_ground_truth(centroids, class_labels, test_Y, test_X))
		homogeneity_list.append(performance_using_homogenity(centroids, class_labels, test_Y, test_X))
		ARI_list.append(performance_using_ARI(centroids, class_labels, test_Y, test_X))
		NMI_list.append(performance_using_NMI(centroids, class_labels, test_Y, test_X))
		calinski_list.append(performance_using_calinski_harabasz(centroids, class_labels, test_Y, test_X))

	get_plot(ground_truth_list, K_list, 'ground_truth.jpeg', 1, 'K', 'ground truth accuracy', 'ground_truth VS K')
	get_plot(homogeneity_list, K_list, 'homogeneity.jpeg', 2, 'K', 'homogeneity score', 'homogeneity VS K')
	get_plot(ARI_list, K_list, 'ARI.jpeg', 3, 'K', 'ARI score', 'ARI VS K')
	get_plot(NMI_list, K_list, 'NMI.jpeg', 4, 'K', 'NMI score', 'NMI VS K')
	get_plot(silhouette_list, K_list, 'silhouette.jpeg', 5, 'K', 'silhouette score', 'silhouette VS K')
	get_plot(calinski_list, K_list, 'calinski.jpeg', 6, 'K', 'calinski score', 'calinski VS K')

	best_k, average_accuracy_wang = wangs_method(train_XY_data, 4)
	get_plot(average_accuracy_wang, K_list, 'wang.jpeg', 7, 'K', 'wangs method', 'wangs VS K')

	return silhouette_list.index(max(silhouette_list)) + 1


if __name__ == '__main__' :
	train_XY_data = import_data('ILPD.csv')
	train_X, train_Y, test_X, test_Y = normalize_data(train_XY_data)

	# Question 1  and 2
	K_value = int(input("Enter the value of K: "))

	# centroids, class_labels = Kmeans_clustering(train_X, train_Y, K_value, 5)
	# performance_using_homogenity(centroids, class_labels, test_Y, test_X)
	# performance_using_ground_truth(centroids, class_labels, test_Y, test_X)
	# performance_using_ARI(centroids, class_labels, test_Y, test_X)
	# performance_using_NMI(centroids, class_labels, test_Y, test_X)


	# Plots for Question 2 and 3
	best_K_value = get_plot_for_indexes(train_XY_data)
	print("Best K:", best_K_value)

	# Question 4
	# test_A_kmeans(K_value, train_XY_data)
	# test_A_kplusplus(K_value, train_XY_data)



