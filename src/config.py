import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

path_szb_commons = 'C:/Users/szb37/My Drive/Work efforts/szb_commons/'

### Graphics settings
save_PNG = True
save_SVG = False
plt.rcParams.update({'font.family': 'arial'})
title_fontdict = {'fontsize': 20, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 16, 'fontweight': 'bold'}
ticklabel_fontsize = 14
sns.set_style("darkgrid")
errorbar='sd'
err_kws={'capsize': 3, 'elinewidth': 0.65,'capthick': 0.65}


### Convert dose guess confidence to SD
thr_dose = 10.5 # if the guessed dose is >= thr_dos, then binary guess is 'T', C otherwise 
doseguess_x = np.linspace(0, 30, 500)
conf_to_sd = { # np.linspace(0.5, 6.75, 7)
    1: 6.75, 
    2: 5.708, 
    3: 4.667, 
    4: 3.625, 
    5: 2.583, 
    6: 1.542, 
    7: 0.5}

### Get distirbtuion of confidences from the POP data
pop_master = pd.read_csv("C:\\Users\\szb37\\My Drive\\Projects\\POP\\codebase\\exports POP\\pop_master.csv")
confs = pop_master.loc[(pop_master.measure=='TRTGUESS_pt_conf') & (pop_master.tp=='A0')].score
