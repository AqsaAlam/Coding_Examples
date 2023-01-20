########## PURPOSE ##########

The purpose of this directory is to demonstrate some examples of misc. code I've written.

########## CONTENTS ##########

> yeast_systematic_name_to_full_description.py

The purpose of this file is to convert systematic gene names to long descriptions.

I usually use this file when I have systematic gene names in my representation that I want
to convert to long descriptions so that when I visualize my heat map in Java Treeview, I get
a sense of what kind of function a cluster of genes might be enriched for.

The code requires three files (all of which I've put in here):
- fix_me.txt
- fixed.txt
- full_yeast_descriptions.txt

You first paste your yeast systematic names in fix_me.txt, run the code, and the fixed output 
will be in fixed.txt.

> generate_fasta_files_3_prime_utr.py

The purpose of this file is to download yeast 3'UTR sequences from Ensembl.
The output of this is a FASTA file for each S. cerevisiae gene containing the promoter
sequence for that gene and its orthologous sequences (for up to 36 different yeast). 

I've put an example of one of the outputs (YER024W.txt). 

> diffraction.py

This is a fun file where I model diffraction. It outputs the image of the grating and the
image of what the diffraction is predicted to look like. This doesn't need any other files
to run it, so it can be run as is. Try it out, it's fun!

########## MORE INFO ##########

There is a lot of data that backs up many of these files, so for now I chose to exclude the data
and only showcase the "main code" (which means they can't be run as is, but you can take a look
at what it does). 

