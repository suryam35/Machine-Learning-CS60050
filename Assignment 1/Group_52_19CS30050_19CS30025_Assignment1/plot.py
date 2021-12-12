import numpy as np
import math
import matplotlib.pyplot as plt
from graphviz import Digraph
from train import *
from predict import *


def get_plot(Y, X, file_name, i, xlabel, ylabel, name):

	'''
	This fucntion takes the value of X and Y and plot these values
	on a graph and saves it to a specified file

	Parameters
	----------
	Y:          values of Y coordinate of the points to be plotted 
	X:          values of X coordinate of the points to be plotted
	file_name:  file where finally the plot is exported
	i:          the figure number
	xlabel:     x axis label of the plot
	ylabel:     y axis label of the plot
	name:       name of the plot

	'''

	plot1 = plt.figure(i)
	plt.plot(X, Y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(name)
	plt.savefig(file_name)

def count_nodes(root):
	'''
	This function is used to count the number of nodes present in the tree.

	Parameters
	----------
	root:       the root of tree whose nodes count is required
	
	Returns
	-------
	number_of_nodes: number of nodes in the tree.
	'''
	number_of_nodes = 1
	if root.left != None:
		number_of_nodes += count_nodes(root.left)
	if root.right != None:
		number_of_nodes += count_nodes(root.right)
	return number_of_nodes

def get_depth_and_number_of_nodes_plot(train_X, train_Y, test_X, test_Y):
	'''
	This function is used to plot two important plots
	1) Accuracy vs depth
	2) Accuracy vs nodes

	Parameters
	----------
	train_X:    the training dataset on which the tree is constructed
	train_Y:    the target values for the training dataset
	test_X:     the testing dataset on which the accuracies are measured
	test_Y:     the target values for the testing dataset

	'''
	accuracy_gini_index = []
	accuracy_information_gain = []
	nodes_gini_index = []
	nodes_information_gain = []
	depths = [i for i in range(1, 51)]
	for depth in range(1, 51):
		root_using_gini_index = construct_tree_using_gini_index(train_X, train_Y, 0, depth, 0)
		root_using_information_gain = construct_tree_using_information_gain(train_X, train_Y, 0, depth, 0)
		accuracy_using_gini_index = check_accuracy(root_using_gini_index, test_X, test_Y)
		accuracy_using_information_gain = check_accuracy(root_using_information_gain, test_X, test_Y)
		accuracy_gini_index.append(accuracy_using_gini_index)
		accuracy_information_gain.append(accuracy_using_information_gain)
		nodes_gini_index.append(count_nodes(root_using_gini_index))
		nodes_information_gain.append(count_nodes(root_using_information_gain))
		print("Accuracy-Gini:" , accuracy_using_gini_index, "Depth:" , depth)
		print("Accuracy-Information_Gain:" , accuracy_using_information_gain, "Depth:" , depth)
	get_plot(accuracy_gini_index, depths, 'gini_index_accuracy.jpeg', 1, 'Depth' , 'Gini Accuracy' , 'Accuracy vs Depth')
	get_plot(accuracy_information_gain, depths, 'information_gain_accuracy.jpeg', 2, 'Depth' , 'Information Gain Accuracy' , 'Accuracy vs Depth')
	get_plot(accuracy_gini_index, nodes_gini_index, 'gini_index_nodes.jpeg', 3, 'Nodes' , 'Gini Index Accuracy' , 'Accuracy vs Nodes')
	get_plot(accuracy_information_gain, nodes_information_gain, 'information_gain_nodes.jpeg', 4, 'Nodes' , 'Information Gain Accuracy' , 'Accuracy vs Nodes')


def is_leaf(node):
	'''
	This function checks if the node passed as the parameter
	is a leaf or non-leaf node.

	Parameters
	----------
	node:       the node for which the information is required
	
	Returns
	-------
	returns a bool, true is node is leaf, false otherwie
	'''
	return node.left == None and node.right == None

def get(node):
	'''
	This function is used to get the characteristics of a particular node 
	such as the predicted class of that node, the best attribute chosen 
	to split at that node and the best threshold on which the split was done. 
	These characteristics are used while printing the tree.

	Parameters
	----------
	node:       the node for which all the above mentioned info is required

	Returns
	----------
	a string containing all the information based on whether the node is leaf 
	or non-leaf
	'''
	if is_leaf(node):
		return "Predicted_Class = {}".format(node.predicted_class)
	return "Predicted_Class = {}\nFeature_Index = {}\nThreshold = {} ".format(node.predicted_class, node.feature_index, node.threshold)


def print_tree(root, filename_):
	'''
	This function prints the tree with the help of external library graphviz

	Parameters
	----------
	root:       the root of tree to be printed
	filename_:  name of the file where tree is to be stored in .pdf format
	
	'''
	graph = Digraph('Decision Tree', filename=filename_)
	graph.attr(rankdir='LR', size='1000,500')

	# border of the nodes is set to rectangle shape
	graph.attr('node', shape='rectangle')

	# Do a breadth first search and add all the edges
	# in the output graph
	q = [root]  # queue for the bradth first search
	while len(q) > 0:
		node = q.pop(0)
		if node.left != None:
			graph.edge(get(node), get(node.left), label='< Threshold')
			q.append(node.left)
		if node.right != None:
			graph.edge(get(node), get(node.right), label='>= Threshold')
			q.append(node.right)

	# save file name :  filename.gv.pdf
	graph.render('./'+filename_, view=True)
