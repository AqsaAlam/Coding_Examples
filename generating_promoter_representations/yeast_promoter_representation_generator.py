"""
The purpose of this file is to generate a representation of Z-scores for yeast promoter sequences.

Input: 
    Motifs:
        YETFASCO_MOTIFS_expanded.npz
        REVERSE_YETFASCO_MOTIFS_expanded.npz
        YETFASCO_MOTIFS_expanded_(number from 0-99).npz
        REVERSE_YETFASCO_MOTIFS_expanded_(number from 0-99).npz
    DNA:
        Path to one-hot encoded DNA files containing yeast + orthologous promoter sequences 


"""

# IMPORTS
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU number
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import keras
import tensorflow as tf



# ------------------- THE MODEL FOR THE TRUE MEAN ----------------------

fileF = np.load("YETFASCO_MOTIFS_expanded.npz", "r", encoding = "bytes")
fileR = np.load("REVERSE_YETFASCO_MOTIFS_expanded.npz", "r", encoding = "bytes")

F_motifs = []
R_motifs = []
for value in fileF.values():
    F_motifs.append(value)
for value in fileR.values():
    R_motifs.append(value)
lst_motifs = []
for i in range(len(F_motifs)):
    lst_motifs.append(F_motifs[i])
    lst_motifs.append(R_motifs[i])

MF = np.dstack(lst_motifs)
MF = np.expand_dims(MF, axis=0)
model = tf.keras.Sequential()
INPUT_SHAPE = (500, 4) # width, channels # the sequence # (500, 4)
conv_layerF = tf.keras.layers.Conv1D(filters = 488, trainable = False, kernel_size = 29, padding = "valid", use_bias=False, input_shape=INPUT_SHAPE)
model.add(conv_layerF)
conv_layerF.set_weights(MF)
model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_first"))
model.add(tf.keras.layers.MaxPooling1D(pool_size=472)) 
model.add(tf.keras.layers.Flatten())
intermediate_model_true = tf.keras.Model(inputs=model.inputs, outputs=model.layers[3].output)


# ------------------- THE MODELS FOR THE SCRAMBLED MEAN ----------------------

"""
The purpose of this section is to load the scrambled matrices into a CNN for scanning.
Note: this is split into 4 pieces: 0-25, 25-50, 50-75, 75-100.
This is to prevent memory issues with the GPU.

This code is repetitive and can be improved by making it a function instead...
But it does work as is, so I don't want to touch it right now.
"""

"""
This first segment is for making 4 different lists for embedding into 4 different CNNs.
The motifs are in the list like FWD1, REV1, FWD2, REV2, FWD3, REV3...FWDN, REVN
This is so we can max pool stride 2 over forward and reverse motifs.
"""

F_motifs1 = []
R_motifs1 = []
lst_motifs1 = []


for i in range(0, 25):

    fileF = np.load("SCRAMBLED/FORWARD/YETFASCO_MOTIFS_expanded_" + str(i) + ".npz", "r", encoding = "bytes")
    fileR = np.load("SCRAMBLED/REVERSE/REVERSE_YETFASCO_MOTIFS_expanded_" + str(i) + ".npz", "r", encoding = "bytes")


    for value in fileF.values():
        F_motifs1.append(value)
    for value in fileR.values():
        R_motifs1.append(value)
    for i in range(len(F_motifs1)):
        lst_motifs1.append(F_motifs1[i])
        lst_motifs1.append(R_motifs1[i])
    F_motifs1 = []
    R_motifs1 = []
            
MF1 = np.dstack(lst_motifs1)
MF1 = np.expand_dims(MF1, axis=0)


F_motifs2 = []
R_motifs2 = []
lst_motifs2 = []

for i in range(25, 50):

    fileF = np.load("SCRAMBLED/FORWARD/YETFASCO_MOTIFS_expanded_" + str(i) + ".npz", "r", encoding = "bytes")
    fileR = np.load("SCRAMBLED/REVERSE/REVERSE_YETFASCO_MOTIFS_expanded_" + str(i) + ".npz", "r", encoding = "bytes")


    for value in fileF.values():
        F_motifs2.append(value)
    for value in fileR.values():
        R_motifs2.append(value)
    for i in range(len(F_motifs2)):
        lst_motifs2.append(F_motifs2[i])
        lst_motifs2.append(R_motifs2[i])
    F_motifs2 = []
    R_motifs2 = []
    
MF2 = np.dstack(lst_motifs2)
MF2 = np.expand_dims(MF2, axis=0)

F_motifs3 = []
R_motifs3 = []
lst_motifs3 = []

for i in range(50, 75):

    fileF = np.load("SCRAMBLED/FORWARD/YETFASCO_MOTIFS_expanded_" + str(i) + ".npz", "r", encoding = "bytes")
    fileR = np.load("SCRAMBLED/REVERSE/REVERSE_YETFASCO_MOTIFS_expanded_" + str(i) + ".npz", "r", encoding = "bytes")

    for value in fileF.values():
        F_motifs3.append(value)
    for value in fileR.values():
        R_motifs3.append(value)
    for i in range(len(F_motifs3)):
        lst_motifs3.append(F_motifs3[i])
        lst_motifs3.append(R_motifs3[i])
    F_motifs3 = []
    R_motifs3 = []
            
MF3 = np.dstack(lst_motifs1)
MF3 = np.expand_dims(MF3, axis=0)

F_motifs4 = []
R_motifs4 = []
lst_motifs4 = []


for i in range(75, 100):

    fileF = np.load("SCRAMBLED/FORWARD/YETFASCO_MOTIFS_expanded_" + str(i) + ".npz", "r", encoding = "bytes")
    fileR = np.load("SCRAMBLED/REVERSE/REVERSE_YETFASCO_MOTIFS_expanded_" + str(i) + ".npz", "r", encoding = "bytes")

    for value in fileF.values():
        F_motifs4.append(value)
    for value in fileR.values():
        R_motifs4.append(value)
    for i in range(len(F_motifs4)):
        lst_motifs4.append(F_motifs4[i])
        lst_motifs4.append(R_motifs4[i])
    F_motifs4 = []
    R_motifs4 = []
MF4 = np.dstack(lst_motifs1)
MF4 = np.expand_dims(MF4, axis=0)


"""
In this second section, we define 4 different models for each chunk of scrambled matrices.
Again, this splitting is only to prevent memory issues.
Theoretically this can just be one model.
"""

# --------------------------------------- MODEL 1 -----------------------------------
model1 = tf.keras.Sequential()
INPUT_SHAPE1 = (500, 4) # width, channels # the sequence # (500, 4)
conv_layerF1 = tf.keras.layers.Conv1D(filters = len(lst_motifs1), trainable = False, kernel_size = 29, padding = "valid", use_bias=False, input_shape=INPUT_SHAPE1)
model1.add(conv_layerF1)
conv_layerF1.set_weights(MF1)
model1.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_first"))
model1.add(tf.keras.layers.MaxPooling1D(pool_size=472))
model1.add(tf.keras.layers.Flatten())
dense1 = tf.keras.layers.Dense(3202, activation='sigmoid') # kernel_constraint="NonNeg", activity_regularizer=l1(0.001))
model1.add(dense1)

# (Define an intermediate model)
intermediate_model1 = tf.keras.Model(inputs=model1.inputs, outputs=model1.layers[3].output)
# --------------------------------------- END MODEL -----------------------------------

# --------------------------------------- MODEL 2 -----------------------------------

model2 = tf.keras.Sequential()
INPUT_SHAPE2 = (500, 4) # width, channels # the sequence # (500, 4)
conv_layerF2 = tf.keras.layers.Conv1D(filters = len(lst_motifs2), trainable = False, kernel_size = 29, padding = "valid", use_bias=False, input_shape=INPUT_SHAPE2)
model2.add(conv_layerF2)
conv_layerF2.set_weights(MF2)
model2.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_first"))
model2.add(tf.keras.layers.MaxPooling1D(pool_size=472))
model2.add(tf.keras.layers.Flatten())
dense2 = tf.keras.layers.Dense(3202, activation='sigmoid') # kernel_constraint="NonNeg", activity_regularizer=l1(0.001))
model2.add(dense2)

# (Define an intermediate model)
intermediate_model2 = tf.keras.Model(inputs=model2.inputs, outputs=model2.layers[3].output)

# --------------------------------------- END MODEL -----------------------------------


# --------------------------------------- MODEL 3 -----------------------------------

model3 = tf.keras.Sequential()
INPUT_SHAPE3 = (500, 4) # width, channels # the sequence # (500, 4)
conv_layerF3 = tf.keras.layers.Conv1D(filters = len(lst_motifs2), trainable = False, kernel_size = 29, padding = "valid", use_bias=False, input_shape=INPUT_SHAPE2)
model3.add(conv_layerF3)
conv_layerF3.set_weights(MF3)
model3.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_first"))
model3.add(tf.keras.layers.MaxPooling1D(pool_size=472))
model3.add(tf.keras.layers.Flatten())
dense3 = tf.keras.layers.Dense(3202, activation='sigmoid') # kernel_constraint="NonNeg", activity_regularizer=l1(0.001))
model3.add(dense3)

# (Define an intermediate model)
intermediate_model3 = tf.keras.Model(inputs=model3.inputs, outputs=model3.layers[3].output)

# --------------------------------------- END MODEL -----------------------------------


# --------------------------------------- MODEL 4 -----------------------------------

model4 = tf.keras.Sequential()
INPUT_SHAPE4 = (500, 4) # width, channels # the sequence # (500, 4)
conv_layerF4 = tf.keras.layers.Conv1D(filters = len(lst_motifs2), trainable = False, kernel_size = 29, padding = "valid", use_bias=False, input_shape=INPUT_SHAPE2)
model4.add(conv_layerF4)
conv_layerF4.set_weights(MF4)
model4.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_first"))
model4.add(tf.keras.layers.MaxPooling1D(pool_size=472))
model4.add(tf.keras.layers.Flatten())
dense4 = tf.keras.layers.Dense(3202, activation='sigmoid') # kernel_constraint="NonNeg", activity_regularizer=l1(0.001))
model4.add(dense4)

# (Define an intermediate model)
intermediate_model4 = tf.keras.Model(inputs=model4.inputs, outputs=model4.layers[3].output)

# --------------------------------------- END MODEL -----------------------------------

# Path to the one-hot encoded DNA sequences.
path = "one_hot_encoded_homolog_npz_files_ATGC_only_KEYS_CHANGED/"


def z_score_representation_generator(output_file_name):
    """
    Input: the name of the output file (whatever you want the output to be called)
    Output: Z-score representation:
            rows are yeast gene systematic names (that the promoter is associated with)
            columns are yeast TF systematic names
    """

    output_z = open(output_file_name, "w")
    
    # get the motif names
    names = []
    for key in fileF:
        names.append(key)
    
    # put the motif names in the file
    for name in names:
        output_z.write("\t" + name)
    output_z.write("\n")
    
    # access the gene file
    for file_name in os.listdir(path):
        
        # open the gene file
        file = np.load(path + file_name, encoding = "bytes")
        
        # inside the file are homologs; loop through the homologs
        homolog_reps_true = []
        homolog_reps1 = []
        homolog_reps2 = []
        homolog_reps3 = []
        homolog_reps4 = []
        names = []
        for homolog_name, homolog_seq in file.items():
            

            if homolog_seq.shape == (1, 500, 4):
                names.append(homolog_name)
                
                # generate the true representation vector
                rep_true = intermediate_model_true(homolog_seq)
                homolog_reps_true.append(rep_true)
                
                # generate all the 244x1 representation vectors
                
                rep1 = intermediate_model1(homolog_seq)
                rep2 = intermediate_model2(homolog_seq)
                rep3 = intermediate_model3(homolog_seq)
                rep4 = intermediate_model4(homolog_seq)
                
                # store them in lists
                
                homolog_reps1.append(rep1)
                homolog_reps2.append(rep2)
                homolog_reps3.append(rep3)
                homolog_reps4.append(rep4)
                    

                    
                
        # average across all TFs to generate a new representation
        # each big rep contains 25 little reps for a total of 100 reps 
        
        
        # TRUE MEAN
        
        gene_rep_true_mean = np.mean(np.array(homolog_reps_true), axis = 0)
        # MEANS
        
        gene_rep1_mean = np.mean(np.array(homolog_reps1), axis = 0)
        gene_rep2_mean = np.mean(np.array(homolog_reps2), axis = 0)
        gene_rep3_mean = np.mean(np.array(homolog_reps3), axis = 0)
        gene_rep4_mean = np.mean(np.array(homolog_reps4), axis = 0)
        
        
        
        # Split each one into lists of tinier representations
        
        # MEANS
        
        list_rep1_mean = np.array_split(gene_rep1_mean[0], 25)
        list_rep2_mean = np.array_split(gene_rep2_mean[0], 25)
        list_rep3_mean = np.array_split(gene_rep3_mean[0], 25)
        list_rep4_mean = np.array_split(gene_rep4_mean[0], 25)
        

        # add them to the same list
        
        # MEANS
        # Now I have a list of 100 244x1 arrays of means 
        
        all_reps_mean = list_rep1_mean + list_rep2_mean + list_rep3_mean + list_rep4_mean
        
    
        #Average the reps over all 100 iterations 
        
        # MEAN AND STDEV FOR MEANS
        
        average_over_100_mean_mean = np.mean(np.array(all_reps_mean), axis = 0)
        average_over_100_mean_stdev = np.std(np.array(all_reps_mean), axis = 0)
        
        z_scores = (gene_rep_true_mean[0] - average_over_100_mean_mean)/(average_over_100_mean_stdev + 0.01)
    
        
        # put the gene name in the file
        output_z.write(">" + file_name.strip(".npz"))
    
    
        for number in z_scores:
            output_z.write("\t" + str(number))
        output_z.write("\n")
        
    
    output_z.close()


# Generate the representation
file_name = "toy_yeast_promoter_representation.txt"
z_score_representation_generator(file_name)
