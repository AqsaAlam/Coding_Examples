########## PURPOSE ##########

The purpose of these three directories:

> HUMAN PROMOTER
> YEAST 3'UTR
> YEAST PROMOTER

is to generate a representation for human promoters, yeast 3'UTRs, and yeast promoters.

########## HOW IT WORKS ##########

Each example is ready to run as is, with 5 one-hot encoded DNA files to run the model on.
The output of each model will have 5 rows and n columns, where n is the number of motifs.

If you want to run the representation properly, change the path in the code so it points to the
directory of one-hot encoded DNA sequences. 

If you want to have different motifs, change the path in the code so it points to the directory
of your motifs.

Within each directory, I've put the .npz files (for sequences and motifs) necessary to run the toy example.
So, you can run it as is.

########## MORE INFO ##########

If you want to see how the .npz files were generated, go to the DATA folder (not the CODES folder).
The data is there as well as the python files which converted it to .npz files.
