"""
This python file contains the class node for the construction of the dcision tree 
from the input dataset and  forms decision tree out of it using the ID3 
algorithm and using gini index and information gain

"""

import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from graphviz import Digraph
from predict import *
from train import *
from plot import *



if __name__ == '__main__':
	train_XY_data = import_data("heart.dat")
	train_X, train_Y, test_X, test_Y = get_validation_and_train_data(train_XY_data,100)

	print("The solution for Question 1")
	for depth in range(1,20):
	  root_using_gini_index = construct_tree_using_gini_index(train_X, train_Y, 0, depth, 0)
	  root_using_information_gain = construct_tree_using_information_gain(train_X, train_Y, 0, depth, 0)
	  accuracy_using_gini_index = check_accuracy(root_using_gini_index, test_X, test_Y)
	  accuracy_using_information_gain = check_accuracy(root_using_information_gain, test_X, test_Y)
	  print("Accuracy-Gini:" , accuracy_using_gini_index, "Depth:" , depth)
	  print("Accuracy-Information_Gain:" , accuracy_using_information_gain, "Depth:" , depth)

	print("\n\nThe solution for Question 2")

	best_root, average_accuracy_gini_index, average_accuracy_information_gain = get_best_root(train_XY_data)
	print("Average accuracy using gini index: ", average_accuracy_gini_index)
	print("Average accuracy using information gain", average_accuracy_information_gain)
	print_tree(best_root, 'decision_tree_old.gv')
	
	train_X, train_Y, test_X, test_Y = get_validation_and_train_data(train_XY_data,501)

	print("\n\nThe solution for Question 3")
	get_depth_and_number_of_nodes_plot(train_X, train_Y, test_X, test_Y)

	tree_root_gini = construct_tree_using_gini_index(train_X, train_Y, 0, 6, 0)
	print("Accuracy_Gini_Tree:" , check_accuracy(tree_root_gini, test_X, test_Y), "Nodes: ", count_nodes(tree_root_gini))
	tree_root_info = construct_tree_using_information_gain(train_X, train_Y, 0, 6, 0)
	print("Accuracy_Gini_Tree:" , check_accuracy(tree_root_info, test_X, test_Y), "Nodes: ", count_nodes(tree_root_info))
	print_tree(tree_root_gini, 'decision_tree_gini.gv')
	print_tree(tree_root_info, 'decision_tree_info.gv')

	print("\n\nThe solution for Question 4 & 5")
	print("Accuracy_before_pruning:" , check_accuracy(best_root, test_X, test_Y), "Nodes: ", count_nodes(best_root))

	new_best_root = prune(best_root, best_root, test_X, test_Y)
	print("Accuracy_after_pruning:" , check_accuracy(new_best_root, test_X, test_Y), "Nodes: ", count_nodes(new_best_root))
	print_tree(new_best_root, 'decision_tree_new.gv')
 
	
	
	
	
# Accuracy_original: 0.9444444444444444 Nodes:  87
# Accuracy_new: 0.9629629629629629 Nodes:  37
