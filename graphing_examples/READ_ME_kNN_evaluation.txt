########## PURPOSE ##########

The purpose of this directory is to generate the "regression over evolution" figure and nucleotide/protein
sequence similarity baselines for that figure.

########## HOW IT WORKS ##########

First, this was run to generate representations at each clade (no need to run again):

> yeast_promoter_representation_increasing_clades.py

To generate the plot that shows regression over evolution, just run this:

> kNN_evaluation_over_clades.py

(You will have to manually label protein and nucleotide blast baselines. Top line is protein, bottom line
is nucleotide, on both graphs).

To generate the numbers for the sequence similarity baselines (no need to run again), run:

> kNN_evaluation_blastn_blastp.py

########## MORE INFO ##########

None of these need to be run again; they're just to show how the figures were generated. 

I fought for my life doing the sequence similarity analysis TT_TT

The nucl_stress.csv, nucl_deleteome.csv, prot_stress.csv, and prot_deleteome.csv files were generated
by doing an all-by-all BLAST against only those genes that were in the analysis (about 2500?). That way
every gene will have a neighbour that exists in all datasets. 

pls trust me... TT_TT
