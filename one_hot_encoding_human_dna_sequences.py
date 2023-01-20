"""
This file will one-hot encode each FASTA file and save it as a .npz file.
This file is for human.
"""

# IMPORTS
import os
from Bio import SeqIO
import numpy as np

    
# ==========================================================================

"""
CONVERT TO NUMPY ARRAY FILES, FILTERING NNN SEQUENCES
"""


def one_hot_encode(sequence):
    """
    This code will one-hot encode a DNA sequence.
    It's a bit inefficient but it does the job...
    """

        A = []
        T = []
        C = []
        G = []
        for n in sequence:
            if n == "A":
                A.append(1)
                T.append(0)
                C.append(0)
                G.append(0)
            elif n == "T":
                A.append(0)
                T.append(1)
                C.append(0)
                G.append(0)
            elif n == "C":
                A.append(0)
                T.append(0)
                C.append(1)
                G.append(0)
            elif n == "G":
                A.append(0)
                T.append(0)
                C.append(0)
                G.append(1)

        # The order is A, C, G, T!
        transposed_matrx = list(zip(*[A, C, G, T]))
        for row in range(len(transposed_matrx)):
            transposed_matrx[row] = list(transposed_matrx[row])
        return np.array(transposed_matrx)

                

path = "all_fasta_files"
count = 0
for file_name in os.listdir(path):
    d = {}
    file = open(path + "/" + file_name, "r")
    for SeqRecord in SeqIO.parse(file, "fasta"):
        # In the most inefficient way possible, exclude those sequences that have anything other than ATGC
        if str(SeqRecord.seq).find("N") == -1 and str(SeqRecord.seq).find("Y") == -1 and str(SeqRecord.seq).find("W") == -1 and str(SeqRecord.seq).find("S") == -1 and str(SeqRecord.seq).find("R") == -1 and str(SeqRecord.seq).find("M") == -1 and str(SeqRecord.seq).find("K") == -1:
            count += 1
            # One hot encode the matrix
            matrx = one_hot_encode(str(SeqRecord.seq))
            # This expand_dims expands the dimension by 1 for loading into the encoder
            matrx = np.expand_dims(matrx, axis=0)
            # Associate the key (name of gene associated with promoter) with the value (one-hot encoded sequence)
            d[str(SeqRecord.id)] = matrx
    # I only want to keep those files that have at least H. sapiens + 9 orthologous species = 10 sequences
    if count >= 10:
            # If there are 10 or more species in the file, save the .npz file
            np.savez(file_name.strip(".txt") + ".npz", **d)
    count = 0
    file.close()



