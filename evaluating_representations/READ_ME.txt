########## PURPOSE ##########

The purpose of this code is to be able to show an example of code I wrote to evaluate representations.

I wrote it to evaluate any representation against any dataset I wanted to perform the regression on.
(For example, on various gene expression datasets)

########## HOW IT WORKS ##########

(Note: this is written the way it is because I wrote these instructions so my supervisor could redo 
these steps and evaluate his own representations).

TO RUN REPRESENTATION_EVALUATOR

Dependencies: numpy, pandas, scipy, sklearn, matplotlib, sys, re

$ conda activate tf_gpu 
(Does not require GPUs, it's just that this 
environment already has all the dependencies installed)

$ python kNN_EVALUATE_REPRESENTATIONS.py [file_to_evaluate_representations_on].txt [representation_file_1].txt [representation_file_2].txt ... [representation_file_3].txt

Example:
$ python kNN_EVALUATE_REPRESENTATIONS.py yeast_GO_categories.txt reverse_homology.txt yeast_promoter_evolution.txt naive_method.txt
(Please keep file names short...)
(All files should have the first column named "Genes", with SYSTEMATIC gene names ONLY)
(Columns can be anything)
(Please no nan values)

Files outputted:

summary_of_R2_scores_for_[name_of_representation].txt (LIST OF ALL R² VALUES FROM HIGHEST TO LOWEST)
distribution_of_R2_values.png (BOX PLOTS)
distribution_of_R2_values_outliers_removed.png (BOX PLOTS WITHOUT OUTLIERS)

Example output with explanation in brackets:

(NAME OF THE REPRESENTATION):
Now evaluating the following representation:
yeast_promoter_evolution 

(NUMBER OF R² SCORES ASSOCIATED WITH COLUMNS IN THE EVALUATION THAT WERE NAN, SO DROPPED):
2 out of 321 R² score(s) were dropped because they were nan

(NAMES OF COLUMNS IN EVALUATION TEXT FILE THAT WERE DROPPED)
The following columns were dropped:
Dsc E3 ubiquitin ligase complex
mitotic spindle

(NUMBER OF COLUMNS IN THE EVALUATION IN THIS ANALYSIS):
There are n = 319 R² value(s) in this analysis.

(MEAN AND STANDARD DEVIATION OF R²):
The average R² is 0.01210 +/- 0.04352

(TOP 5 R² CATEGORIES):
The top 5 R² categories are:
0.49929 MCM complex
0.34625 chaperonin-containing T-complex
0.32085 mitochondrial respiratory chain complex IV
0.21401 proteasome storage granule
0.12865 cellular amino acid metabolic process

########## MORE INFO ##########

It requires a few things to make it run, so you can't run it, but please appreciate how it's written.

I've put an example of the two boxplots that it outputs (which kind of get cut off at the bottom...
I've yet to fix that). 
