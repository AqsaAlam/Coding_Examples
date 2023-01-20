########## PURPOSE ##########

The purpose of this directory is to provide some examples of code where I've had to graph things.

########## HOW IT WORKS ##########

> kNN_evaluation_over_clades.py

The purpose of this code is to generate the "regression over evolution" figure and nucleotide/protein
sequence similarity baselines for that figure. This is a figure in my thesis. 

It required a few files before it to run this one: something to generate a representation with 
a variable number of species (the files of which are not currently in this directory), and another
code that generated the BLASTp and BLASTn "baselines". 

> distributions_and_qq_plots.py 

The purpose of this code is to generate some of the first figures in my paper, the ones that talk about
how the method words to "solve" problems like futility theorem and normalization.

It required a file before it to generate the .npz files used in this code (which are not currently
in this directory).

########## MORE INFO ##########

As the files required to run these are not in the directory, please use these files to simply take
a look at how I typically code graphs. 

The output of the code is in the folder as .png files.
