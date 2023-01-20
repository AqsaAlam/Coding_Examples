########## PURPOSE ##########

The purpose of this directory is to convert systematic gene names to long descriptions.

I usually use this file when I have systematic gene names in my representation that I want
to convert to long descriptions so that when I visualize my heat map in Java Treeview, I get
a sense of what kind of function a cluster of genes might be enriched for.

########## HOW IT WORKS ##########

First, paste your systematic gene names (human or yeast) in the following file:

> fix_me.txt

Then, if you want to convert yeast names, run:

> yeast_systematic_name_to_full_description.py

If you want to convert human names, run:

> human_systematic_name_to_full_description.py

Your output will be located in:

> fixed.txt

Additionally, this file:

> motif_gene_names.xlsx 

is just a file associating yeast motif systematic names with their common names.
In Sheet2, the third column has the format SystematicName_CommonName.
I copy and paste this to replace the header in my representation file, 
THEN I cluster in Cluster 3.0 and visualize in Java Treeview. 

########## MORE INFO ##########

The descriptions for yeast are located in:

> full_yeast_descriptions

and the descriptions for human are located in:

> full_human_descriptions

Anything that is not a yeast or human gene name will not return an error;
instead, it will be left as is. 

