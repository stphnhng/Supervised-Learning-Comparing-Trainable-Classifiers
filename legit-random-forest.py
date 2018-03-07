'''
    Stephen Hung
    PseudoCode: http://proceedings.mlr.press/v28/denil13-supp.pdf
                https://onlinecourses.science.psu.edu/stat857/node/181

    Gini Impurity:
        https://datascience.stackexchange.com/questions/10228/gini-impurity-vs-entropy
        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
'''
import csv
import sys
import random
from math import sqrt
import time

STRING_TO_INT = {} # dictionary containing the string to integer values.

'''
    load data as a list.
    Param:
        filename = name of the data in directory
'''
def load_data(filename): # loads data as a list.
    dataset = list()
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            dataset.append(row)
    return dataset

'''
    Converts all strings (classes) into integer representations
    Param:
        dataset = data to alter.
'''
def convert_dataset_strings(dataset):
    # Convert strings into float representation in the dataset.
    global STRING_TO_INT

    int_representation = 0
    for row in dataset:
        for index in range(0, len(row)):
            try:
                row[index] = float(row[index])
            except Exception:
                if(row[index] not in STRING_TO_INT): # increasing int representation for different items (0,1,2,...)
                    STRING_TO_INT[row[index]] = int_representation
                    row[index] = STRING_TO_INT[row[index]]
                    int_representation+=1
                else:
                    row[index] = STRING_TO_INT[row[index]]
    return dataset

'''
    Creates a random sample set from dataset with sample size of sample_size
    Chooses from dataset with replacement.

    Param: 
        dataset = data to choose sample from 
        sample_size = size of the sample being chosen
'''
def random_sample(dataset, sample_size):
    sample = list()
    while len(sample) < sample_size:
        # choose from data with replacement
        index = random.randrange(len(dataset))
        # without replacement:
        '''
        index = random.randrange(len(dataset))
        dataset.pop(index)
        '''
        sample.append(dataset[index])
    return sample



'''
    Calculate the Gini Impurity for a set of items (group)
        i is in {1,2,.... # of classes}
        p_i = fraction of items labeled with class i in data
        Gini Impurity = 1- sum(i = 1 to # of classes) p_i ^2

    Param:
        total_rows = total amount of rows in complete dataset - used for weighting
        group = a group of data to be tested for gini purity
        classes = The different classifications
'''
def gini_impurity(total_rows, group, classes):
    num_classes = len(classes)
    impurity_score = 0
    if len(group) == 0:
        return 0.0
    for i in classes:
        class_i_count = 0
        for row in group:
            if(row[-1] == i):
                class_i_count+=1
        # fraction of items with class i
        p_i = class_i_count * 1.0 / len(group) 
        impurity_score += (p_i)*(p_i)
    # 1 - impurity_score is the result
    # multiply by len(group) / total_rows in order to weight each impurity score based
    # on the size of the given group. (prevent biases)
    return (1 - impurity_score) * (1.0 * len(group) / total_rows)

'''
    split's the entire training set into two groups,
        one less than the given value and
        one greater than and equal to the given value
    the given value is each row's chosen indices (the randomly chosen predictors)

    Param:
        dataset = the data to be split into two on row[feature_index] and value
        feature_index = the predictor index being tested
        value = the value used to split the data (< on left, >= on right)
'''
def split_dataset_on_index(dataset, feature_index, value):
    split1 = list()
    split2 = list()
    for row in dataset:
        if row[feature_index] < value:
            split1.append(row)
        else:
            split2.append(row)
    return split1, split2

'''
    Finds the best possible split point through gini impurity

    Param:
        dataset = data to find the best split point for
        num_predictors = number of predictors to be used (bouned cause of RF)
'''
def find_split_point(dataset, num_predictors):
    # want to find minimum, so set to sys.maxsize as default
    best_gini = sys.maxsize
    best_feature_index = sys.maxsize
    best_split_value = sys.maxsize
    best_split_groups = None


    list_of_classes = set() # set because it disallows duplicates
    for row in dataset:
        list_of_classes.add(row[-1])
    list_of_classes = list(list_of_classes) # make it a list to make life easier

    # (2.) take random sample without replacement of the predictors

    # length of row without class at the end
    length_of_row = len(dataset[0]) - 1

    class_features = list() # random list of class features of size num_predictors
    while(len(class_features) < num_predictors):
        feature_index = random.randrange(length_of_row)
        if feature_index not in class_features:
            class_features.append(feature_index)

    # (3.) Construct a split by using predictors selected in Step 2
    # go through each row in dataset
    for row in dataset:
        # select the randomly chosen predictors and split dataset into two using the row's predictor 
        # and find highest purity split
        for feature_index in class_features:
            # check every row's feature index and see if it is a good split point
            split_value = row[feature_index]
            # split dataset into two based on the split_value (< on left) and (>= on right)
            split1, split2 = split_dataset_on_index(dataset, feature_index, split_value)
            # total length of the dataset
            total_rows = float(len(split1)) + float(len(split2))
            # get the gini value of this split
            split_gini_value = gini_impurity(total_rows, split1, list_of_classes) + gini_impurity(total_rows, split2, list_of_classes)

            # find best possible (minimum) gini value - high purity thus less inequality among groups.
            if split_gini_value < best_gini:
                best_gini = split_gini_value
                best_feature_index = feature_index
                best_split_value = split_value
                best_split_groups = (split1, split2)

    # returns these values for use in actually splitting decision tree 
    # does not return best_gini since this was only used to find the best split - we have already found so no need.
    return {'feature_index': best_feature_index, 'split_value': best_split_value, 'split_groups': best_split_groups}

'''
    returns the class with the highest count in group.

    Param: 
        group = the data from which the highest count class will be found.
'''
def terminal_node(group):
    all_group_classes = [row[-1] for row in group]
    return max(set(all_group_classes), key = all_group_classes.count)

'''
    Split the tree until all leaf nodes are found.

    Param:
        node = the root of the tree about to be split
        num_predictors = number of predictors to be used in find_split_point
'''
def split_tree(node, num_predictors):
    left_node, right_node = node['split_groups']
    del(node['split_groups']) # no more use for it now

    # If the nodes(groups) are empty, then set it to 
    if not left_node or not right_node:
        node['left'] = node['right'] = terminal_node(left_node + right_node)
        return

    # if there are only 1 or less elements in the list, get a leaf node
    if len(left_node) <= 1:
        node['left'] = terminal_node(left_node)
    else:
        node['left'] = find_split_point(left_node, num_predictors)
        split_tree(node['left'], num_predictors)

    # if there are only 1 or less elements in the list, get a leaf node
    if len(right_node) <= 1:
        node['right'] = terminal_node(right_node)
    else:
        node['right'] = find_split_point(right_node, num_predictors)
        split_tree(node['right'], num_predictors)


'''
    (4.) Repeat 2 & 3 until tree is large

    Constructs a Decision tree using the training dataset and bounded predictors

    Param:
        training_dataset = the dataset used for making the decision tree
        num_predictors = number of predictors to be used in find_split_point
'''
def construct_decision_tree(training_dataset, num_predictors):
    # root is the start of the training dataset - i.e first split
    root = find_split_point(training_dataset, num_predictors)
    split_tree(root, num_predictors)
    return root

'''
    Traverses the tree to find out what the tree's prediction will be given the data: row

    Param:
        tree_node = the root of the passed in tree, will be used for tree traversal
        row = The row of data from which the class will be predicted
'''
def tree_traversal(tree_node, row):
    # if less than, then go down the left path
    if( row[tree_node['feature_index']] < tree_node['split_value'] ):
        # if it is a dictionary this means it is still not a leaf node - keep going
        if isinstance(tree_node['left'], dict):
            return tree_traversal(tree_node['left'], row)
        else:
            return tree_node['left']
    else:
        # is >= value, go down right path
        if isinstance(tree_node['right'], dict):
            return tree_traversal(tree_node['right'], row)
        else:
            return tree_node['right']

'''
    Predict the data from the out of bag sample's row.

    Param:
        trees = The decision trees created
        dataset_row = the row of data to be predicted 
'''
def predict_outofbag(trees, dataset_row):
    # predict a result for each decision tree
    class_predictions = list()
    for decision_tree in trees:
        class_predictions.append(tree_traversal(decision_tree, dataset_row))
    # returns class with highest count
    '''
        (7.) for each observation, count the number of trees that is classified in one category

        (8.) assign each observation to a final category by a majority vote
    '''
    return max(set(class_predictions), key=class_predictions.count)

'''
    (6.) Repeat 1 - 5 a large amount of times

    The Random Forest Algorithm, constructs num_trees decision trees and then gets the 
    classification predictions from the test_data using those trees

    Param:
        dataset: the complete dataset to base RF on
        test_data: the test dataset which will be predicted by the decision trees
        num_trees: Number of trees to be created
        train_sample_ratio: the ratio of training data to complete data
        num_predictors: amount to bound the number of predictors that can be used in RF
'''
def random_forest_alg(dataset, test_data, num_trees, train_sample_ratio, num_predictors):
    class_predictions = list() # list of predictions for each test_data's row of data
    tree_list = list() # list of trees
    length_of_sample = round(train_sample_ratio * len(dataset))
    for i in range(num_trees):
        training_sample = random_sample(dataset, length_of_sample)
        tree_root = construct_decision_tree(training_sample, num_predictors)
        tree_list.append(tree_root)
    # (5.) Drop the out-of-bag data down the tree - store class assigned to each observation
    # Could have been done in previous for loop but that would result in O(n^2) complexity,
    # this is better (O(n))
    for row in test_data:
        class_predictions.append(predict_outofbag(tree_list, row))
    return class_predictions

# n_samples = target values (class labels)
# n_features = features of a class

'''
    PseudoCode: (ctrl+f) to find in code
    (1.) Take a random sample of size N with replacemenet from the data (bootstrap sample)

    (2.) Take a random sample without replacement of the predictors

    (3.) Construct a split by using predictors selected in Step 2

    (4.) Repeat 2 & 3 until tree is large

    (5.) Drop the out-of-bag data down the tree - store class assigned to each observation

    (6.) Repeat 1 - 5 a large amount of times

    (7.) for each observation, count the number of trees that is classified in one category

    (8.) assign each observation to a final category by a majority vote
'''

def main():
    filename = 'ecoli.csv'
    print('Enter the data file you would like to use:')
    file_input = input('0 for default, a string otherwise:   ')
    if file_input != '0':
        filename = file_input
    print('\n')
    dataset = load_data(filename)
    dataset = convert_dataset_strings(dataset)

    # clean data set so there are no empty rows
    if [] in dataset:
        dataset.remove([])

    # (1.) get test data
    # Sample Size - Size of Test data to be used to identify accuracy of Classifier
    #             - Default is 50% of Total Dataset
    length_of_sample = round(0.3 * len(dataset))
    test_data = random_sample(dataset, length_of_sample)

    # 2 Main Tuning Parameters
    # Number of Trees - Number of Decision Trees
    #                 - 500 is default
    num_trees = 129
    print('Enter the number of trees you would like to use:')
    num_tree_input = input('(0 for default, an integer > 0 otherwise:   ')
    if(num_tree_input != '0'):
        num_trees = int(num_tree_input)
    print('\n')
    # Number of Predictors Sampled - number of predictors sampled at each split
    #                              - Sampling 2 -5 each time is often adequate
    #                              - sqrt is easiest.
    num_predictors = float(sqrt(len(dataset[0])-1))
    bounds = len(dataset[0])-1
    print('Enter the number of predictors you want sampled:')
    num_predictors_input = input('(0 for default or an integer x such that, 0 < x < ' + str(bounds) + ' :   ')
    print('\n')
    if(num_predictors_input != '0'):
        num_predictors = int(num_predictors_input)

    # Start Time for alg timing purposes
    start = time.time()

    # returns list of predictions with each element being the majority vote of all decision tree's prediction
    # for a corresponding row in test_data
    predictions = random_forest_alg(dataset, test_data, num_trees, 0.5, num_predictors)
    
    # End Time for alg timing purposes
    end = time.time()


    # Compare actual values in test_data to predictions made by random forest
    actual_values = [row[-1] for row in test_data]
    correct_values = 0
    for i in range(len(actual_values)):
        if actual_values[i] == predictions[i]:
            correct_values+=1
    # compute algorithm efficiency
    alg_acc = correct_values * 1.0 / len(actual_values) * 100.0
    print('Random Forest Algorithm Accuracy:', end='\t')
    print(alg_acc)
    # time elapsed for random forest alg.
    print('Algorithm Time Elapsed:', end='\t')
    print(str(end-start))

# main method when run as a script
if __name__ == '__main__':
    main()