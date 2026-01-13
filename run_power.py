import src.folders as folders
import src.power as power
import src.config as config
from itertools import product
import pandas as pd
import numpy as np
import os


if False: ### Generate mock trial data with both binary and continous trt guess

    prefix = 'mock_cgr'
    n_trials = 10
    sample = 10

    ### Generate binary guess data

    # Define parameters
    scenario_params = []
    for cgr in np.arange(0.5, 1, 0.05):
        cgr = round(cgr, 2)
        scenario_params.append(
            (f'cgr_{cgr}', 
            {'type': 'binaryguess',
            'arm_params':{
                'C': {'cgr': cgr,},
                'T': {'cgr': cgr,},},}))

    # Get data
    df_patientsData=[]
    for scenario_param in scenario_params:

        scenario = scenario_param[0]
        params = scenario_param[1]

        df_patientData = power.DataGeneration.get_df_patientsData(
            scenario = scenario, 
            n_trials = n_trials, 
            sample = sample, 
            params = [params])
        df_patientsData.append(df_patientData)

    # Concatanate and save 
    df_patientsData = pd.concat(df_patientsData, ignore_index=True)
    df_patientsData.to_csv(os.path.join(folders.power, f'{prefix}_trials.csv'), index=False)
    df_patientsData.head()

if False: ### How confidence effects conf weighted approach

    prefix = 'mock_gmgc'
    n_trials = 50
    sample = 100

    ### Generate binary guess data
    # Define parameters
    scenario_params = []
    for conf in [1, 7]:
        scenario_params.append(
            (f'conf{conf}', 
            {
                'type': 'normal',
                'arm_params':{
                    'C': {'mean': 15, 'sd': 4,},
                    'T': {'mean':  5, 'sd': 4,},                
                },}))

    # Get data
    df_patientsData=[]
    for scenario_param in scenario_params:
        scenario = scenario_param[0]
        params = scenario_param[1]

        df_patientData = power.DataGeneration.get_df_patientsData(
            scenario = scenario, 
            n_trials = n_trials, 
            sample = sample, 
            params = [params])
        df_patientsData.append(df_patientData)

    # Concatanate dfs, rename, clip
    df_patientsData = pd.concat(df_patientsData, ignore_index=True)
    df_patientsData = df_patientsData.rename(columns={'value': 'gmg'})
    df_patientsData['gmg'] = df_patientsData['gmg'].round(1)
    df_patientsData['gmg'] = df_patientsData['gmg'].clip(lower=0, upper=30)

    # Add confidence
    df_patientsData.loc[df_patientsData.scenario=='conf1', 'conf'] = 1 
    df_patientsData.loc[df_patientsData.scenario=='conf7', 'conf'] = 7 
    df_patientsData['conf'] = df_patientsData['conf'].astype(int)

    df_weighted_gmgs, df_combined_weighted_gmgs = power.Helpers.get_df_weighted_gmgs(df_patientsData)

    print(df_combined_weighted_gmgs)

if False: ### How confidence effects conf-to-SE approach

    prefix = 'mock'
    n_trials = 50
    sample = 100

    ### Generate binary guess data
    # Define parameters
    scenario_params = []
    for conf in ['Low', 'High']:
        scenario_params.append(
            (f'conf{conf}', 
            {
                'type': 'normal',
                'arm_params':{
                    'C': {'mean': 15, 'sd': 4,},
                    'T': {'mean':  5, 'sd': 4,},                
                },}))

    # Get data
    df_patientsData=[]
    for scenario_param in scenario_params:
        scenario = scenario_param[0]
        params = scenario_param[1]

        df_patientData = power.DataGeneration.get_df_patientsData(
            scenario = scenario, 
            n_trials = n_trials, 
            sample = sample, 
            params = [params])
        df_patientsData.append(df_patientData)

    # Concatanate dfs, rename, clip
    df_patientsData = pd.concat(df_patientsData, ignore_index=True)
    df_patientsData = df_patientsData.rename(columns={'value': 'gmg'})
    df_patientsData['gmg'] = df_patientsData['gmg'].round(1)
    df_patientsData['gmg'] = df_patientsData['gmg'].clip(lower=0, upper=30)

    # Add confidence & corresponding SEs
    df_patientsData.loc[df_patientsData.scenario=='confLow',  'conf'] = np.random.choice([1,1,1], size=df_patientsData.loc[df_patientsData.scenario=='confLow'].shape[0]) 
    df_patientsData.loc[df_patientsData.scenario=='confHigh', 'conf'] = np.random.choice([7,7,7], size=df_patientsData.loc[df_patientsData.scenario=='confHigh'].shape[0]) 
    df_patientsData['gmg_se'] = df_patientsData['conf'].map(config.conf_to_se)

    ### Calculate combined mean and SE for each scenario, trt - i.e. avg across trials
    scenarios = df_patientsData.scenario.unique()
    trts = df_patientsData.trt.unique()

    rows=[]
    for scenario, trt in product(scenarios, trts):

        df_tmp = df_patientsData.loc[
            (df_patientsData.scenario==scenario) & 
            (df_patientsData.trt==trt)]

        row = {}
        row['scenario'] = scenario
        row['trt'] = trt
        row['comb_gmg'] = df_tmp['gmg'].mean()
        comb_gmg_se = np.sqrt((df_tmp['gmg_se']**2).sum() / df_tmp['gmg_se'].shape[0])
        row['comb_gmg_ciL'] = row['comb_gmg'] - 1.96*comb_gmg_se
        row['comb_gmg_ciH'] = row['comb_gmg'] + 1.96*comb_gmg_se
        row['comb_gmg_moe'] = (row['comb_gmg_ciH'] - row['comb_gmg_ciL']) / 2
        rows.append(row)

    df_combined_gmgs = pd.DataFrame(rows)

    for col in ['comb_gmg', 'comb_gmg_ciL', 'comb_gmg_ciH']:
        df_combined_gmgs[col] = df_combined_gmgs[col].clip(lower=0, upper=30)

    for col in ['comb_gmg', 'comb_gmg_ciL', 'comb_gmg_ciH', 'comb_gmg_moe']:
        df_combined_gmgs[col] = df_combined_gmgs[col].round(1)


    print(df_combined_gmgs)

if True: ### How confidence effects conf-to-SE approach

    prefix = 'mock'
    n_trials = 50
    sample = 100

    ### Generate binary guess data
    # Define parameters
    scenario_params = []
    for gmgdiff in np.arange(0, 10, 2):
        scenario_params.append(
            (f'gmgdiff{gmgdiff}', 
            {
                'type': 'normal',
                'arm_params':{
                    'C': {'mean': (10.5 - gmgdiff/2), 'sd': 4,},
                    'T': {'mean': (10.5 + gmgdiff/2), 'sd': 4,},                
                },}))

    # Get data
    df_patientsData=[]
    for scenario_param in scenario_params:
        scenario = scenario_param[0]
        params = scenario_param[1]

        df_patientData = power.DataGeneration.get_df_patientsData(
            scenario = scenario, 
            n_trials = n_trials, 
            sample = sample, 
            params = [params])
        df_patientsData.append(df_patientData)

    # Concatanate dfs, rename, clip
    df_patientsData = pd.concat(df_patientsData, ignore_index=True)
    df_patientsData = df_patientsData.rename(columns={'value': 'gmg'})
    df_patientsData['gmg'] = df_patientsData['gmg'].round(1)
    df_patientsData['gmg'] = df_patientsData['gmg'].clip(lower=0, upper=30)

    # Add confidence & corresponding SEs
    df_patientsData['conf'] = np.random.choice(config.confs, size=df_patientsData.shape[0]) 
    df_patientsData['gmg_se'] = df_patientsData['conf'].map(config.conf_to_se)

    ### Calculate combined mean and SE for each scenario, trt - i.e. avg across trials
    scenarios = df_patientsData.scenario.unique()
    trts = df_patientsData.trt.unique()

    rows=[]
    for scenario, trt in product(scenarios, trts):

        df_tmp = df_patientsData.loc[
            (df_patientsData.scenario==scenario) & 
            (df_patientsData.trt==trt)]

        row = {}
        row['scenario'] = scenario
        row['trt'] = trt
        row['comb_gmg'] = df_tmp['gmg'].mean()
        comb_gmg_se = np.sqrt((df_tmp['gmg_se']**2).sum() / df_tmp['gmg_se'].shape[0])
        row['comb_gmg_ciL'] = row['comb_gmg'] - 1.96*comb_gmg_se
        row['comb_gmg_ciH'] = row['comb_gmg'] + 1.96*comb_gmg_se
        row['comb_gmg_moe'] = (row['comb_gmg_ciH'] - row['comb_gmg_ciL']) / 2
        rows.append(row)

    df_combined_gmgs = pd.DataFrame(rows)

    for col in ['comb_gmg', 'comb_gmg_ciL', 'comb_gmg_ciH']:
        df_combined_gmgs[col] = df_combined_gmgs[col].clip(lower=0, upper=30)

    for col in ['comb_gmg', 'comb_gmg_ciL', 'comb_gmg_ciH', 'comb_gmg_moe']:
        df_combined_gmgs[col] = df_combined_gmgs[col].round(1)

    print(df_combined_gmgs)