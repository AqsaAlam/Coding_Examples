# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:54:53 2022

@author: aqsaa
"""


# IMPORTS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import tensorflow as tf


# LOAD THE FORWARD AND REVERSE MOTIFS
fileF = np.load("YETFASCO_MOTIFS_expanded.npz", "r", encoding = "bytes")
fileR = np.load("ALANS_CODE_REVERSE_YETFASCO_MOTIFS_expanded.npz", "r", encoding = "bytes")

# PUT THE FORWARD AND REVERSE MOTIFS INTO LISTS
F_motifs = []
R_motifs = []

for value in fileF.values():
    F_motifs.append(value)
    
for value in fileR.values():
    R_motifs.append(value)

    
# THE CODE FOR THE MODEL

# Define the model
model = tf.keras.Sequential()

# Define the input shape (500 bp sequence with ATGC)
INPUT_SHAPE = (500, 4) # width, channels # the sequence # (500, 4)

# Define the cut-off score
SCORE_CUT=6.2

# Initialize the motif layer and add it to the model, then initialize the weights to be the TFs
motif_layer = tf.keras.layers.Conv1D(filters = 488, trainable = False, kernel_size = 29, padding = "same", use_bias=False, input_shape=INPUT_SHAPE)
model.add(motif_layer)
motif_layer.set_weights(MF)

# Add the max pooling to average forward and reverse TFs
fwd_rev_av = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_first")
model.add(fwd_rev_av)

# Make everything below the threshold 0
model.add(tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=SCORE_CUT))



# THEN, OUTSIDE OF THE MODEL, IN YOUR FOR LOOP...

# Call on the model, remove the None dimension with the [0], and convert from a tensor to a numpy array
output_of_model = model(input_sequence)[0].numpy() 

# OUTPUT OF THIS IS (len_sequence, num_motifs) (ex. 500, 244)

# Now we want to change everything above 0 to a 1
output_of_model[output_of_model > 0] = 1

# NOW YOU HAVE A (500, 244) ARRAY. IF IT HELPS, TRANSPOSE IT.
output_of_model = output_of_model.transpose()

# NOW WE CAN LOOP THROUGH THIS (244, 500) OBJECT.
# Each row is a motif scanned across the entire sequence.
# This array has 1s wherever there is a binding site.


