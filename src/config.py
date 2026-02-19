import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

path_szb_commons = 'C:/Users/szb37/My Drive/Work efforts/szb_commons/'

''' General configs '''
### Graphics settings
save_PNG = True
save_SVG = False
plt.rcParams.update({'font.family': 'arial'})
plt.rcParams['figure.dpi'] = 300 # Set display DPI
sns.set_style("darkgrid")

### Use these settings when saving plot as image
title = {'fontsize': 16, 'fontweight': 'bold'}
axislabel = {'fontsize': 12, 'fontweight': 'bold'}
ticklabel = 14

### Use these settings when displaying image in notebook
nb_title = {'fontsize': 8, 'fontweight': 'bold'}
nb_axislabel = {'fontsize': 4, 'fontweight': 'bold'}
nb_ticklabel = 4
nb_legendtitle = {'size': 4, 'weight': 'bold'}
nb_legend = 4

### Errorbar settings
errorbar='sd'
err_kws={'capsize': 3, 'elinewidth': 0.65,'capthick': 0.65}


''' Project specific configs '''
### Misc configs
digits=3 # Number of digits round CIs to
#methods=['cgr', 'bbi', 'gmg', 'gmgc']
methods=['cgr',]
ropes={
    'cgr': 0.14, 
    'bbi': 0.2, 
    'gmg':  5,
    'gmgc': 5,}

### Bayesian stats
n_draws=500
cgr_ranges = [
    (0.000, 0.375),
    (0.375, 0.625),
    (0.625, 1.000),
]


### Convert dose guess confidence to SD
thr_dose = 10.5 # if the guessed dose is >= thr_dos, then binary guess is 'T', otherwise 'C'
doseguess_x = np.linspace(0, 30, 500)
conf_to_se = {# np.linspace(2, 20, 7)/1.96
    1: 5.102*2, 
    2: 4.337*2, 
    3: 3.571*2, 
    4: 2.806*2, 
    5: 2.041*2, 
    6: 1.276*2, 
    7: 0.510*2,}

### Guess confidences from POP
confs = [7.0,
 2.0,
 6.0,
 3.0,
 6.0,
 3.0,
 3.0,
 6.0,
 5.0,
 6.0,
 4.0,
 5.0,
 7.0,
 2.0,
 7.0,
 4.0,
 3.0,
 7.0,
 5.0,
 5.0,
 4.0,
 4.0,
 7.0,
 5.0]