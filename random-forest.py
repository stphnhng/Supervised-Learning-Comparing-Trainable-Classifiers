'''
    Stephen Hung
    Implementation of Random Forest Classifier
    Trained using data from iris.data: https://archive.ics.uci.edu/ml/datasets/iris
    sources:
        http://dni-institute.in/blogs/cart-decision-tree-gini-index-explained/
        https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
        https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html
'''

import csv
import sys
from random import seed
from random import randrange
from math import sqrt

STRING_TO_INT = {} # dictionary containing the string to integer values.

def load_data(filename): # loads data as a list.
    dataset = list()
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            dataset.append(row)
    return dataset

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
    split data set into groups 
'''
def get_split_groups(index, index_value, dataset):
    left_group = list()
    right_group = list()
    # splits data set into two groups depending on given index value
    for row in dataset:
        if row[index] < index_value:
            left_group.append(row)
        else:
            right_group.append(row)
    return (left_group, right_group)

'''
    evaluate a split point using groups from get_split_groups
'''
def get_gini_value(groups, class_values):

    n_instances = float(sum([len(group) for group in groups]))
    # number of instances - sum of length of groups

    gini_value = 0.0

    for group in groups:
        # calculate the gini value
        size = float(len(group)) # length of group
        if size == 0: # size is 0 so ignore group
            continue
        score = 0.0
        # gini value formula
        # http://dni-institute.in/blogs/cart-decision-tree-gini-index-explained/
        for class_val in class_values:
            p = [row[-1] for row in group].count(class_val) / size
            score += (p*p)

        gini_value += (1.0 - score) * (size / n_instances)
    return gini_value

'''
    split decision trees on several split points based on attribute and value of attribute.
    Cost function is the Gini index - finds inequality of distribution (lowest is the most equal)

    find best split point: evaluate cost of each value in training dataset for each input variable

    sample of input attributes chosen randomly and without replacement - 
        each input variable considered once when looking for split point with lowest cost

'''
def get_split(dataset, num_features):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = sys.maxsize, sys.maxsize, sys.maxsize, None
    features = list()

    # get random list of indices for features.
    while len(features) < num_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)

    for index in features: # for each random index in each feature.
        for row in dataset: # split each row of dataset to find best gini value
            # split data set into groups based on random index
            groups = get_split_groups(index, row[index], dataset)
            # calculate gini value from split groups
            gini_value = get_gini_value(groups, class_values)
            # if it is the lowest gini cost so far, it is the best value
            if gini_value < best_score:
                best_index = index
                best_value = row[index]
                best_score = gini_value
                best_groups = groups

    return {'index': best_index, 'value':best_value, 'score':best_score, 'groups':best_groups}


def terminal_node(group):
    outcomes = [row[-1] for row in group]
    # returns the class where the # of classes is the greatest
    return max(set(outcomes), key=outcomes.count)

def split_tree(node, max_depth, min_size, n_features, depth):

    left, right = node['groups'] # left and right groups after being split.
    del(node['groups']) # node['groups'] = None

    if not left or not right: # if left or right is non existent
        node['left'] = node['right'] = terminal_node(left + right)
        # set both left and right as a terminal node (return highest number of classes from both left and right.
        return

    if depth >= max_depth:
        node['left'], node['right'] = terminal_node(left), terminal_node(right)
        # if reached max depth, set left and right node to highest number of classes from left and right node respectively
        return

    
    if len(left) <= min_size:
        node['left'] = terminal_node(left) 
        # if length of left group is smaller than minimum allowed size, set it equal to highest number of classes from left.
    else:
        # if > min_size, then find best split node and split the tree based on that and keep going.
        node['left'] = get_split(left, n_features)
        split_tree(node['left'], max_depth, min_size, n_features, depth+1)

    if len(right) <= min_size:
        # if length of right group is smaller than minimum allowed size, set it equal to highest number of classes from right.
            node['right'] = terminal_node(right)
    else:
        # if > min_size, then find best split node and split the tree based on that and keep going.
        node['right'] = get_split(right, n_features)
        split_tree(node['right'], max_depth, min_size, n_features, depth+1)

def build_decision_tree(train, max_depth, min_size, n_features):
    # build a decision tree
    root = get_split(train, n_features) # root to split at - found by using gini score
    split_tree(root, max_depth, min_size, n_features, 1) # split tree based on root.
    return root

def class_predict(node, row):
    # predicts class based on the values from training decision tree
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict): 
            return class_predict(node['left'], row) # still dictionary, not values keep going
        else:
            return node['left'] # return class value
    else:
        if isinstance(node['right'], dict):
            return class_predict(node['right'], row)
        else:
            return node['right']

def subsample_data(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    # builds random sub sample of data where len(subsample) = len(dataset) * ratio
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def predict_bagging_data(trees, row):
    # predict based on decision trees created in random forest alg and test data.
    predictions = [class_predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def random_forest_alg(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    # random forest algorithm
    trees = list()
    for i in range(n_trees):
        # number of trees
        sample = subsample_data(train, sample_size)
        tree = build_decision_tree(sample, max_depth, min_size, n_features) 
        # build a decision tree based on the sample size of (len(dataset) * ratio) where ratio = sample_size
        # sample data is from the training data
        trees.append(tree) # append decisions trees for each tree
    predictions = [predict_bagging_data(trees, row) for row in test] 
    # bagging - create multiple different trees of training data set with different samples
    # then return predictions for each different bagged tree
    return predictions


def main():
    global NUM_STRING_COLUMN

    filename = 'iris.data'
    dataset = load_data('iris.data')
    dataset = convert_dataset_strings(dataset)

    dataset.remove([])

    total_input_feature = len(dataset[0]) - 1
    num_feature_split = float(sqrt(total_input_feature)) 
    # num of attributes is the bound for the number of features to be considered.
    
    best_split_point_values = get_split(dataset, num_feature_split)


    n_folds = 5
    max_depth = 10
    min_size = 1
    sample_size = 1.0 # percentages of values in a sample dataset from actual data set

    scores = list()

    dataset_split = list() # actually split data set
    dataset_copy = list(dataset) # copy to be used when generating the split data trees
    fold_size = int(len(dataset) / n_folds) # split data set into n_folds parts
    for i in range(n_folds):
        # for each split data part
        fold = list()
        while len(fold) < fold_size:
            # gets random rows to the data set until it is the size of the split data
            index = randrange(len(dataset_copy))
            # gets a random row from one of the parts of the data set
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold) # append the n_folds split data trees
    folds = dataset_split # folds = list of split data sets

    for fold in folds: # for each split data set
        train_set = list(folds) # full data set
        train_set.remove(fold) # remove the split data set from the rest
        train_set = sum(train_set, [])

        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy) # appends each row in split data set
            row_copy[-1] = None


        predicted = random_forest_alg(train_set, test_set, max_depth, min_size, sample_size, 1, num_feature_split)
        # predicts test_set classes based on full training set without test set
        # given max_depth of tree of 10
        # min size of 1
        # sample size of 1
        actual = [row[-1] for row in fold] # - actual values in fold
        correct_values = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct_values+=1 # finds num of correct values from prediction
        accuracy = correct_values / float(len(actual)) * 100.0
        scores.append(accuracy) # scores for each split set

    mean_accuracy = sum(scores) / float(len(scores))

    print('Split into ' + str(n_folds) + ' trees')
    print('Accuracy Scores for each split data set: ', scores)
    print('Mean Accuracy: ', mean_accuracy)



if __name__ == '__main__':
    main()
