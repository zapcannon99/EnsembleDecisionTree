#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:40:33 2019

@author: Tenkichi-MAC
"""
import warnings
warnings.filterwarnings("ignore")

# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import re
import copy
from scipy import stats
import time
import math
import statistics

from decimal import Decimal


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    # raise Exception('Function not yet implemented!')
    
    dictionary = dict()
    index = np.arange(0, len(x))
    values = np.unique(x)
    for value in values:
        dictionary[value] = index[x==value]
        
    return dictionary


def entropy(y, w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    # raise Exception('Function not yet implemented!')
    
    if(w is None):
        raise Exception("oy, w is not supposed to be empty here")
    
    # get # of samples 
    # grab the counts of 0 and 1 (since this is a binary problem of 0 and 1) since entropy calc just needs that
    values = partition(y)
    probs = list()
    total_prob = np.sum(w)
    # compute the log2 for each term

    for v in values:
        hits = values[v]
        probs.append(np.sum(w[hits])/total_prob)
    logs = np.log2(probs)
#    if((logs == -np.inf).any()):
#        print(logs)
#    get the negative summation of the terms
    return -np.sum(np.multiply(probs, logs))



def mutual_information(x, y, w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    
    BIG NOTE: x is actually an array of true and falses. Mutual info works by showing which match attr val pair or not
    """

    # INSERT YOUR CODE HERE
    # raise Exception('Function not yet implemented!')
    
    if(w is None):
        raise Exception("oy, w is not supposed to be empty here")
 
    # get all unique values
    entropy_total = entropy(y, w)
    conditional_entropies = list()
    values = partition(x)
    P_x_total = np.sum(w)
    for value in values:
        hits = values[value]
        y_x = y[hits]
        w_x = w[hits]
        P_x = np.sum(w[hits])/P_x_total
        conditional_entropies.append(P_x * entropy(y_x, w_x))
    mutual_information = entropy_total - np.sum(conditional_entropies)
    return mutual_information


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, w=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
#    print(y)
#    print("depth{}".format(depth))
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # raise Exception('Function not yet implemented!')
    
    # if the atribute_value_pairs have not been created, make it in the first id3 call
    if(attribute_value_pairs is None):
        #if the depth is 0, then we need to populate the attribute_value_pairs
        if(depth == 0):
            d = x[0].size #d being the number of features/attributes
            attribute_value_pairs = list()
            for attr in np.arange(0,d):
                values = np.unique(x[:, attr])
                for value in values:
#                    attribute_value_pairs = np.append(attribute_value_pairs, (attr, value))
                    attribute_value_pairs.append((attr, value))
            attribute_value_pairs = np.array(attribute_value_pairs)
        else:
            # return the label that appears the most
            return stats.mode(y)[0]
    
    if(w is None):
        w = np.repeat(1/len(y), len(y))
    
    values = np.unique(y)
    if(len(values) == 1):
        return values[0]
    
    if(depth == max_depth):
        mode, count = stats.mode(y)
        return mode[0]
    
    #get the mutual informations
    mi = dict()
    for pair in attribute_value_pairs:
        avp = tuple(pair)
#        print("--------------pair {}-----------------".format(pair))
        column = pair[0]
        value = pair[1]
        
        x_col = np.array(x[:, column])
        hits = x_col == value
#        print(pair)
        mi[avp] = mutual_information(hits, y, w)
            
    best_pair = max(mi.keys(), key=(lambda k: mi[k]))
    
    node = dict()
    
    best_column = best_pair[0]
    best_value = best_pair[1]
    best_x = x[:, best_column]
    hits = best_x == best_value
    misses = best_x != best_value 
    unmatched_x = x[misses]
    unmatched_y = y[misses]
    unmatched_w = w[misses]
    
    matched_x = x[hits]
    matched_y = y[hits] 
    matched_w = w[hits]
    
    pair_index = (attribute_value_pairs == best_pair).all(axis=1).nonzero()[0][0]
    new_attribute_value_pairs = np.delete(attribute_value_pairs, pair_index, axis=0)
#    print(best_pair)
#    print(new_attribute_value_pairs)
#    input()
    # First, the branch where attribute_value_pair is true
    true_branch = (best_column, best_value, True)
    if(len(matched_y) == 0):
        modes, counts = stats.mode(y)
        true_branch_value = modes[0]
    else:
        true_branch_value = id3(matched_x, matched_y, new_attribute_value_pairs, depth + 1, max_depth, w = matched_w)
    node[true_branch] = true_branch_value
    
    # Then false
    false_branch = (best_column, best_value, False)
    if(len(unmatched_y) == 0):
        modes, counts = stats.mode(y)
        false_branch_value = modes[0]
    else:
        false_branch_value = id3(unmatched_x, unmatched_y, new_attribute_value_pairs, depth + 1, max_depth, w = unmatched_w)
    node[false_branch] = false_branch_value
    
    return node


def bagging(x, y, max_depth, num_trees):
    ensemble = list()
    for iteration in range(num_trees):
        bootstrap_indexes = np.random.uniform(0, len(y), len(y)).astype(int)
        bs_y = y[bootstrap_indexes]
        bs_x = x[bootstrap_indexes]
        tree = id3(bs_x, bs_y, max_depth=max_depth)
        ensemble.append((1, tree))
    
    return ensemble
    
    
def boosting(x, y, max_depth, num_stumps):
    ensemble = list()
    weights = np.repeat(1/len(y), len(y))
    for iteration in range(num_stumps):
        stump = id3(x, y, max_depth = max_depth, w = weights)
#        print("{}| {}".format(iteration, stump))
        y_pred = np.array([predict_example_single(x_datapoint, stump) for x_datapoint in x])
        error, margin = compute_error(y, y_pred, weights) #weighted error
        alpha = (1/2) * math.log((1-error)/error)
        exp_margin = np.exp(-alpha * margin)
        numerator = np.multiply(weights, exp_margin)
        norm_factor = np.sum(numerator)
        weights = numerator / norm_factor
        
        ensemble.append((alpha, stump))
        
    return ensemble
        
    
def predict_example(x, h_ens):
    votes = dict()
    
    for (a, tree) in h_ens:
        vote = predict_example_single(x, tree)
        if((vote in votes) == False):
            votes[vote] = a
        else:
            votes[vote] = votes[vote] + a
        
    decision = max(votes.keys(), key=(lambda k: votes[k]))
    return decision


def predict_example_single(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if(type(tree) != type(dict())):
        return tree;
    
    for node in tree:
        attr = node[0]
        val = node[1]
        truth = node[2]
        if((x[attr] == val) == truth):
            return predict_example_single(x, tree[node])


def compute_error(y_true, y_pred, w=None):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    
    if(w is None):
        w = np.repeat(1/len(y_true), len(y_true))
    
    #make sure everything is an np.array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # INSERT YOUR CODE HERE
    misses = y_true != y_pred
    error = np.sum(w[misses])/np.sum(w)
    
    margin = np.ones(len(y_true))
    misses = y_true == y_pred
    margin[misses] = -1
    return (error, margin)


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid