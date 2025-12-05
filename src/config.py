import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

path_szb_commons = 'C:/Users/szb37/My Drive/Work efforts/szb_commons/'

### Graphics settings
save_PNG = True
save_SVG = False
plt.rcParams.update({'font.family': 'arial'})
title_fontdict = {'fontsize': 16, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 12, 'fontweight': 'bold'}
ticklabel_fontsize = 14
sns.set_style("darkgrid")
errorbar='sd'
err_kws={'capsize': 3, 'elinewidth': 0.65,'capthick': 0.65}


### Convert dose guess confidence to SD
thr_dose = 10.5 # if the guessed dose is >= thr_dos, then binary guess is 'T', otherwise 'C'
doseguess_x = np.linspace(0, 30, 500)
conf_to_se = { #  np.linspace(2, 20, 7)/1.96
    1: 5.102*2, 
    2: 4.337*2, 
    3: 3.571*2, 
    4: 2.806*2, 
    5: 2.041*2, 
    6: 1.276*2, 
    7: 0.510*2}


### Get distirbtuion of confidences from the POP data
pop_master = pd.read_csv("C:\\Users\\szb37\\My Drive\\Projects\\POP\\codebase\\exports POP\\pop_master.csv")
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