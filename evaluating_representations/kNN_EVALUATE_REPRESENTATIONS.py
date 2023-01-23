# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:04:09 2022

@author: aqsaa

This is for evaluating representations on different datasets (kNN regression)

"""

# IMPORTS

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sys
import re



#################### INPUTS INTO THE MODEL SHOULD BE FILES ####################

# Files given as command line arguments
evaluation_file = sys.argv[1]
list_of_representations = sys.argv[2:]




##################### DEFINE THE REPRESENTATION EVALUATOR #####################

def representation_evaluator(name_of_representation, representation_file, evaluation_file):
    
    """
    INPUTS:
        
        file1, which is the representation
        file2, which is the file you want to evaluate the representations on
        
    OUTPUTS:
        
        A file with an "experiments" columns and the R^2 it's associated with
        A boxplot figure
        Print the number of R^2s that were nan, and what column they're associated with
        Print the mean and standard deviation of the R^2
        
    """
    
    print("")
    print("Now evaluating the following representation:" + "\n" + name_of_representation)
    print("")
    
    
    ################### FIND THE INTERSECTION OF THE DATAFRAMES ###################
    
    # Here's the representation
    df1 = pd.read_csv(representation_file, sep="\t")
    
    # Here's the thing you want to evaluate the representation on (ex. GO categories)
    df2 = pd.read_csv(evaluation_file, sep="\t")
    
    
    # Get the columns of each dataframe
    cols1 = df1.columns
    cols2 = df2.columns
    
    # Find the intersection of the dataframes
    int_df = pd.merge(df1, df2, how ='inner')
    
    # Split the dataframes into two dataframes, one of the representation and one of the evaluation
    # These two representations now have the same genes in the same order
    
    representation = int_df[cols1]
    evaluation = int_df[cols2]
    
 
    # one of the columns will be Genes
    
    # Remove columns not used for predictions
    genes = representation["Genes"]
    del representation["Genes"]
        
    # Set up X (predictor column) 
    X = representation.iloc[:,:].values
    

    
    ############# MAKING A DICTIONARY OF GENE WITH ITS CLASS VECTOR ###############
    
    # Create a dictionary that contains...
    # example for GO categories: {gene : [0 0 0 1 0 0 0 1...0 0 1]}
    # example for gene expression: {gene: [0.23 0.55 0.25 -0.10...0.42 -0.15 0.21]}
    
    evaluation_dictionary = {}
    for row in evaluation.iterrows():
        evaluation_dictionary[row[1]["Genes"]] = np.array(row[1][1::])
        
        
    
    ###################### CALCULATING NEAREST NEIGHBOURS #########################    
    
    
    # Create the distance matrix
    distance_matrix = cdist(X, X, "cosine")
    
    # truthfully I don't remember what this does :sob:
    # I think I wrote this to fix an error I was getting...
        
    distance_matrix[distance_matrix == -np.inf] = 0
    distance_matrix[distance_matrix == +np.inf] = 1
    distance_matrix[~np.isfinite(distance_matrix)] = 0
    
    
    # Fit NearestNeighbors on distance matrix and retrieve neighbors
    knn_distance_based = (
        NearestNeighbors(n_neighbors=1, metric="precomputed")
            .fit(distance_matrix)
    )
    
    # Get a list where each index (ex. gene at index 0) contains the index of its nearest neighbour
    # ex. gene list: [0 1 2 3 4 5...]
    # then, neighbour: [3 4 5 0 1 2...]
    
    nn_1 = knn_distance_based.kneighbors(return_distance=False)
    # annoyingly, these seem to be like [[3],[4],[5],[0],[1],[2]] instead...
    # let's fix that:
    nn_1 = [val[0] for val in nn_1]
    
    
    print("There are " + str(len(nn_1)) + " genes in each computation of R^2") 
    
    
    ################ MAKING ARRAYS FOR NEAREST NEIGHBOUR PREDICTIONS ##############
    
    gene_evaluations = []
    neighbour_evaluations = []
    
    for i_t, i_n in enumerate(nn_1):    # i_t is the index of the true gene, neighbour is the index of the neighbour
        gene = genes[i_t] # gene name of gene
        neighbour = genes[i_n] # gene name of neighbour
        
        # Convert gene_evaluations and neighbour_evaluations to dtype = float
        # Because for some DUMB reason they're dtype = object...why python, why...
        
        gene_evaluation = np.array(evaluation_dictionary[gene], dtype = float) # evaluation vector of gene
        neighbour_evaluation = np.array(evaluation_dictionary[neighbour], dtype = float) # evaluation vector of neighbour
        
        gene_evaluations.append(gene_evaluation) # put the evaluation vector for the gene in its list
        neighbour_evaluations.append(neighbour_evaluation) # put the evaluation vector for the neighbour in its list
    
    
    
    
    
    ##################### MAKE A TRUE DF AND A PREDICTED DF #######################

    
    # Convert them to numpy arrays (and they need to be transposed for some reason)
    gene_evaluations = np.array(gene_evaluations).transpose()
    neighbour_evaluations = np.array(neighbour_evaluations).transpose()
    
    # Get the evaluation headers/categories
    evaluation_categories = cols2.tolist()[1::]
    
    # Turn them into a predicted data frame and a true data frame
    true_class_df = pd.DataFrame.from_dict(dict(zip(evaluation_categories, gene_evaluations)))
    pred_class_df = pd.DataFrame.from_dict(dict(zip(evaluation_categories, neighbour_evaluations)))






    ######################### COMPUTE THE R^2 VALUES ##########################

    # These are for us to see how many true/prediction columns generate nan R^2 values
    nan_counter = 0
    nan_columns = []
    total_columns = len(true_class_df.columns) # for comparison
    
    # These are so we can generate box plots and a file of R^2 values later
    R2_scores = []
    columns = []

    # Loop through the columns; we want to generate an R^2 value for each column
    for column in true_class_df.columns:
        
        # Grab the "true vector" and the "prediction vector"
        true_vector = true_class_df[column]
        pred_vector = pred_class_df[column]
        
        # We don't want any R^2s that are nan, which happens when the standard deviation
        # is 0, so let's just get rid of them...but count them.
        if np.std(true_vector) == 0 or np.std(pred_vector) == 0:
            nan_counter += 1
            nan_columns.append(column)
            
        # Otherwise, we want to compute the R^2
        else:
            corr_matrix = np.corrcoef(true_vector.to_numpy(), pred_vector.to_numpy())
            corr = corr_matrix[0,1]
            R2_score = corr**2
    
    
        # Then, append the score to a growing list of R^2 values
            R2_scores.append(R2_score)
            columns.append(column)



    ######################### PRINT REQUIRED OUTPUTS ##########################

    # How many R2 scores were dropped because they were nan?
    print(str(nan_counter) + " out of " + str(total_columns) + " R\u00b2 score(s) were dropped because they were nan")
    if not nan_counter == 0:
        print("")
        print("The following columns were dropped:")
        for column in nan_columns:
            print(column)
        
    
    # Only take those columns and those R2 scores that are not nan
    columns = [columns[i] for i in range(len(R2_scores)) if str(R2_scores[i]) != 'nan']
    R2_scores = [x for x in R2_scores if str(x) != 'nan']
    
    print("")
    print("There are n = " + str(len(R2_scores)) + " R\u00b2 value(s) in this analysis.")
    print("")
    print("The average R\u00b2 is " + str.format('{0:.5f}', np.mean(R2_scores)) + " +/- " + str.format('{0:.5f}', np.std(R2_scores)))



    R2_scores, columns = map(list, zip(*sorted(zip(R2_scores, columns), reverse=True)))


    # Save R2 scores to a file
    summary_of_R2_scores = open("summary_of_R2_scores_for_" + name_of_representation + ".txt", "w")
    summary_of_R2_scores.write("Column_Name" + "\t" + "R^2" + "\n")  
    for i in range(len(R2_scores)):
        summary_of_R2_scores.write(str(columns[i]) + "\t" + str(R2_scores[i]) + "\n")
    summary_of_R2_scores.close()
    
    
    print("")
    print("The top 5 R\u00b2 categories are:")
    for i in range(5):   
        print(str.format('{0:.5f}', R2_scores[i]), columns[i])
        
        
        
    # Finally, return the list of R2_scores for plotting
    return R2_scores
        


############################ GENERATE THE BOX PLOT ############################
    
    
# Evaluate each representation to get a distribution of R^2 values
data = []
labls = []
for representation in list_of_representations:
    
    # Get the name of the representation
    name_of_representation = str(representation).split("\\")[len(str(representation).split("\\"))-1].strip(".txt")
    name_of_representation = name_of_representation.split("/")[len(name_of_representation.split("/"))-1]
    labls.append(name_of_representation)
    
    # Evaluate the representation
    x = representation_evaluator(name_of_representation, representation, evaluation_file)
    data.append(x)
    
    print("")
    print("###################################################################")


# Squeeze the labels so they're not super long
labls_new = [re.sub("(.{10})", "\\1\n", labl, 0, re.DOTALL) for labl in labls]

# Add the n = ___ to each label
for i, labl in enumerate(labls_new):
    labls_new[i] = labl + "\n(n = " + str(len(data[i])) + ")"

# Plot the representations on a single box plot
fig1 = plt.figure()
plt.boxplot(data)
plt.title("Distribution of R\u00b2 Values for Each Representation")
plt.xlabel("Name of Representation")
plt.ylabel("R\u00b2")
plt.xticks(np.arange(len(labls))+1, labls_new)
fig1.savefig("distribution_of_R2_values.png")   
    

fig2 = plt.figure()
plt.boxplot(data, showfliers=False)
plt.title("Distribution of R\u00b2 Values for Each Representation\n(Outliers Removed)")
plt.xlabel("Name of Representation")
plt.ylabel("R\u00b2")
plt.xticks(np.arange(len(labls))+1, labls_new)
fig2.savefig("distribution_of_R2_values_outliers_removed.png")
    
