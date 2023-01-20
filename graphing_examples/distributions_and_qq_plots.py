# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:38:16 2022

@author: aqsaa
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

ev = np.load("HSF1_TOD6_GIS1_MSN2_LYS14_evol_av_thom_removed.npz", encoding = "bytes")

for key, value in ev.items():
    if key == "HSF1":
        HSF1 = value
    elif key == "TOD6":
        TOD6 = value
    elif key == "GIS1":
        GIS1 = value
    elif key == "MSN2":
        MSN2 = value
    elif key == "LYS14":
        LYS14 = value


bins=np.histogram(np.hstack((HSF1, TOD6, MSN2)), bins=70)[1] #get the bin edges


fig, ax = plt.subplots(2)
ax[0].hist(HSF1, color='dodgerblue', alpha=0.6, bins=bins, ec='k', label = "Hsf1")
ax[0].hist(TOD6, color='gold', alpha=0.6, bins=bins, ec = 'k', label = "Tod6")
ax[0].hist(MSN2, color='indianred', alpha=0.6, bins=bins, ec = 'k', label = "Mns2")
ax[0].legend()
ax[0].set(xlabel="Best Match Score to TFBM (bits)")



ev = np.load("HSF1_TOD6_GIS1_MSN2_LYS14_evol_z_scores_thom_removed.npz", encoding = "bytes")

for key, value in ev.items():
    if key == "HSF1":
        HSF12 = value
    elif key == "TOD6":
        TOD62 = value
    elif key == "GIS1":
        GIS12 = value
    elif key == "MSN2":
        MSN22 = value
    elif key == "LYS14":
        LYS142 = value
        

bins=np.histogram(np.hstack((HSF12, TOD62, MSN22)), bins=70)[1] #get the bin edges


ax[1].hist(HSF12, color='dodgerblue', alpha=0.6, bins=bins, ec='k', label = "Hsf1")
ax[1].hist(TOD62, color='gold', alpha=0.6, bins=bins, ec = 'k', label = "Tod6")
ax[1].hist(MSN22, color='indianred', alpha=0.6, bins=bins, ec = 'k', label = "Msn2")
ax[1].legend()
ax[1].set(xlabel="Z Score")
fig.show()


xy = [i for i in range(-5, 4337)]
plt.style.use('seaborn')


#create Q-Q plot with 45-degree line added to plot
fig, ax = plt.subplots(1, 3, figsize = (9, 3))
sm.qqplot(HSF1, line = '45', markersize = 2, markerfacecolor="dodgerblue", label = "Before Normalization", ax=ax[0])
sm.qqplot(HSF12, line = '45', markersize = 2, markerfacecolor='grey', label = "After Normalization", ax=ax[0])
ax[0].plot(xy, xy, color = "black", linewidth = 2.5)
sm.qqplot(TOD6, line ='45', markersize = 2, markerfacecolor="gold", label = "Before Normalization", ax=ax[1])
sm.qqplot(TOD62, line ='45',  markersize = 2, markerfacecolor="grey", label = "After Normalization", ax=ax[1])
ax[1].plot(xy, xy, color = "black", linewidth = 2.5)
sm.qqplot(MSN2, line ='45', markersize = 2, markerfacecolor="indianred", label = "Before Normalization", ax=ax[2])
sm.qqplot(MSN22, line ='45', markersize = 2, markerfacecolor="grey", label = "After Normalization", ax=ax[2])
ax[2].plot(xy, xy, color = "black", linewidth = 2.5)
plt.tight_layout()
plt.show()








