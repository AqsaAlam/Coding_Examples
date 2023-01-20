########## PURPOSE ##########

The purpose of these three Python files

> evol_av_of_best_match.py
> HSF1_RKM3_evolution.py
> distributions_and_qq_plots.py

is to generate some of the first figures in the paper, the ones that talk about
how the method words to "solve" problems like futility theorem and normalization.

########## HOW IT WORKS ##########

The following files does not need to be run again:

> evol_av_of_best_match.py

This Python file generated the following two files:

> HSF1_TOD6_GIS1_MSN2_LYS14_evol_av.npz
> HSF1_TOD6_GIS1_MSN2_LYS14_evol_z_scores.npz
(or the ones with _thom_removed, where I removed a species that wasn't yeast).

I use those files when running:

> distributions_and_qq_plots.py

Which makes the distribution and qq-plots. 

You can also run:

> HSF1_RKM3_evolution.py

which generates a scatter plot and a distribution plot which can be put together in PowerPoint
to generate the figure that talks about how our method addresss futility theorem.

########## MORE INFO ##########

None of these need to be run again; they're just to show how the figures were generated. 
