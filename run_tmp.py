
### Import & setup
import sys
sys.path.append('C:\\Users\\szb37\\My Drive\\Projects\\ADAPT\\ADAPT codebase\\')
import src.folders as folders
import src.power as power
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['figure.dpi'] = 300  # Set display DPI
plt.style.use('seaborn-v0_8-notebook')  # notebook-optimized

prefix = 'mock'
n_trials = 5
sample = 10

### Generate CGR mock data & calc CIs, significance
df_patientsData=[]
ciL = 0.80
ciH = 0.90

for tID in np.arange(0, n_trials, 1):

    cgr = round(np.random.uniform(low=ciL, high=ciH), 3)
    scenario = f'Sample:{tID} CGR:{cgr}'
    params = {
        'type': 'binaryguess',
        'arm_params':{
            'C': {'cgr': cgr},
            'T': {'cgr': cgr},},}

    df = power.DataGeneration.get_df_patientsData(
        scenario = scenario, 
        n_trials = 1, 
        sample = sample, 
        params = [params])
    df['trial'] = tID
    df_patientsData.append(df)

df_patientsData = pd.concat(df_patientsData, ignore_index=True)
df_patientsData.to_csv('tmp.csv', index=False)