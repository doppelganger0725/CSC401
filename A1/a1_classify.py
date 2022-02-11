#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from re import X
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    total_case = np.sum(C)
    total_correct = np.trace(C)
    return total_correct / total_case


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = []
    col_sum = np.sum(C, axis = 0)
    for i in range(len(C)):
        result.append(C[i,i] / col_sum[i])
    return result

def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = []
    row_sum = np.sum(C, axis = 1)
    for i in range(len(C)):
        result.append(C[i,i] / row_sum[i])
    return result
    

classify_name_list = ["SGDClassifier", "GaussianNB", "RandomForestClassifier", "MLPClassifier", "AdaBoostClassifier" ]


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    # print('TODO Section 3.1')
    iBest = None
    acc_max = 0
    classifier = None
    #calculate matrix 
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i in range(5):
            classifier_name = classify_name_list[i]
            if i == 0:
                classifier = SGDClassifier()
            if i == 1:
                classifier = GaussianNB()
            if i == 2:
                classifier = RandomForestClassifier(n_estimators =10, max_depth=5)
            if i == 3:
                classifier = MLPClassifier(alpha = 0.05)
            if i == 4:
                classifier = AdaBoostClassifier()

            classifier.fit(X_train,y_train) 
            y_prediction = classifier.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_prediction)

            acc = accuracy(conf_matrix)
            rec = recall(conf_matrix)
            pre = precision(conf_matrix)


        # For each classifier, compute results and write the following output:
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in pre]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
            if acc > acc_max:
                iBest = i

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    indexes = [1000, 5000, 10000, 15000, 20000]
    classifier = None
    if iBest == 0:
        classifier = SGDClassifier()
    if iBest == 1:
        classifier = GaussianNB()
    if iBest == 2:
        classifier = RandomForestClassifier(n_estimators =10, max_depth=5)
    if iBest == 3:
        classifier = MLPClassifier(alpha = 0.05)
    if iBest == 4:
        classifier = AdaBoostClassifier()
    
    
    
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        for index in indexes:
            num_train = index
            x_amount = X_train[:index]
            y_amount = y_train[:index] 
            
            classifier.fit(x_amount,y_amount) 
            y_prediction = classifier.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_prediction)

            acc = accuracy(conf_matrix)
            
            outf.write(f'{num_train}: {acc:.4f}\n')
    X_1k = X_train[:1000]
    y_1k = y_train[:1000]
    return (X_1k, y_1k)


# Read feature.txt map the index with feature name
feat_dict={}
feat_dict[0] = "Number of tokens in uppercase (≥ 3 letters long)"
feat_dict[1] = "Number of first-person pronouns"
feat_dict[2] = "Number of second-person pronouns"
feat_dict[3] = "Number of third-person pronouns"
feat_dict[4] = "Number of coordinating conjunctions"
feat_dict[5] = "Number of past-tense verbs"
feat_dict[6] = "Number of future-tense verbs"
feat_dict[7] = "Number of commas"
feat_dict[8] = "Number of multi-character punctuation tokens"
feat_dict[9] = "Number of common nouns"
feat_dict[10] = "Number of proper nouns"
feat_dict[11] = "Number of adverbs"
feat_dict[12] = "Number of wh- words"
feat_dict[13] = "Number of slang acronyms"
feat_dict[14] = "Average length of sentences, in tokens"
feat_dict[15] = "Average length of tokens, excluding punctuation-only tokens, in characters"
feat_dict[16] = "Number of sentences"
feat_dict[17] = "Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms"
feat_dict[18] = "Average of IMG from Bristol, Gilhooly, and Logie norms"
feat_dict[19] = "Average of FAM from Bristol, Gilhooly, and Logie norms"
feat_dict[20] = "Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms"
feat_dict[21] = "Standard deviation of IMG from Bristol, Gilhooly, and Logie norms"
feat_dict[22] = "Standard deviation of FAM from Bristol, Gilhooly, and Logie norms"
feat_dict[23] = "Average of V.Mean.Sum from Warringer norms"
feat_dict[24] = "Average of A.Mean.Sum from Warringer norms"
feat_dict[25] = "Average of D.Mean.Sum from Warringer norms"
feat_dict[26] = "Standard deviation of V.Mean.Sum from Warringer norms"
feat_dict[27] = "Standard deviation of A.Mean.Sum from Warringer norms"
feat_dict[28] = "Standard deviation of D.Mean.Sum from Warringer norms"


def add_dict(feature, file):
    '''
    compose feat name dict
    '''
    index = 29
    f = open(file,'r')
    lines = f.readlines()
    for line in lines:
        feature[index] = line.rstrip()
        index += 1
    return feature

# full_feat = add_dict(feat_dict,"/u/cs401/A1/feats/feats.txt")
full_feat = add_dict(feat_dict,"/Users/Ryan Fu/Desktop/fushihh1/A1/a1_feats/feats/feats.txt")



def five_feature(X_train, X_test, y_train, y_test, iBest):
    classifier = None
    if iBest == 0:
        classifier = SGDClassifier()
    if iBest == 1:
        classifier = GaussianNB()
    if iBest == 2:
        classifier = RandomForestClassifier(n_estimators =10, max_depth=5)
    if iBest == 3:
        classifier = MLPClassifier(alpha = 0.05)
    if iBest == 4:
        classifier = AdaBoostClassifier()
    
    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(X_train, y_train)
    X_new_test = selector.transform(X_test)
    temp = selector.pvalues_
    print("\n\n\pvalues\n\n")
    print(temp)
    index = np.argsort(temp)[:5]
    classifier.fit(X_new,y_train) 
    y_prediction = classifier.predict(X_new_test)
    conf_matrix = confusion_matrix(y_test, y_prediction)

    acc = accuracy(conf_matrix)

    return acc, index


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for
        # that number of features:
        feat_num = [5,50]

        for feat in feat_num:
            k_feat = feat
            selector = SelectKBest(f_classif, k=k_feat)
            X_new = selector.fit_transform(X_train, y_train)
            temp = selector.pvalues_
            p_values = []
            index = np.argsort(temp)[:k_feat]
            for j in index:
                p_values.append(temp[j]) 
            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')

        accuracy_1k, feature_index_1k = five_feature(X_1k, X_test, y_1k, y_test, i)
        accuracy_full, feature_index_full= five_feature(X_train, X_test, y_train, y_test, i)
        
        feature_intersection = intersection(feature_index_1k,feature_index_full)
        
        top_5 = feature_index_full

        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')
        


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
        pass


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    #train_test_split
    
    data =  np.load(args.input)
    output = args.output_dir
    lst = data["arr_0"]
    X = []
    y = []
    for i in range(len(lst)):
        X.append(lst[i][:173])
        y.append(lst[i][173])
    # TODO: load data and split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # iBest = class31(output, X_train, X_test, y_train, y_test)
    iBest = 4
    # X_1k, y_1k = class32(output, X_train, X_test, y_train, y_test,iBest)
    X_1k = X_train[:1000]
    y_1k = y_train[:1000]
    print(len(feat_dict))
    class33(output, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(output, X_train, X_test, y_train, y_test, iBest)
    # TODO : complete each classification experiment, in sequence.
    
