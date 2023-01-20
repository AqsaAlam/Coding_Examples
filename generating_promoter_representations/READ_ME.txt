########## PURPOSE ##########

The purpose of these files is to provide some examples of code.

########## HOW IT WORKS ##########

> simple_motif_scanner.py

This is a well-annotated code describing how I used machine learning tools (keras, tensorflow, 
machine learning architecture) to create a fast motif scanner. 

This one in particular has been edited so the output is binary (1 if there is a binding site,
0 if there isn't). 

> yeast_promoter_representation_generator.py

This cannot currently be run as is as it requires many files which are currently not in the directory:
- 100 scrambled forward motifs
- 100 scrambeld reverse motifs
- a file of real forward motifs
- a file of real reverse motifs
- 4337 FASTA files of yeast promoter sequences

However, this is the MAIN CODE used to generate representations of yeast promoter sequences.

I have put an example of the output for this code file in the directory. 

########## MORE INFO ##########

One major improvement to this model that I would like to implement in the future is instead of 
splitting the model into 4 separate models, to write a function that does the splitting for me.

I coded this before I learned the rule that "if you're coding it more than once, write a function
for it". The 4 times repeat of things in this code is to prevent memory issues on the GPU.
