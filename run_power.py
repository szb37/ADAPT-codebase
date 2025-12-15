import src.config as config
import src.power_calc as power
import src.folders as folders
import pandas as pd
import os
import importlib
importlib.reload(power)

prefix='tmp'

if True: ### Generate mock data

    n_trials = 5
    sample_size = 100
    df_trialsData=[]

    scenario = 'tmp'
    df_trialData = power.DataGeneration.get_df_trialsData_params(
        scenario = scenario,
        n_trials = n_trials, 
        sample_size = sample_size, 
        params_cont={
            'mean_C': 10.5, 
            'mean_T': 10.5,        
            'sd': 7.6, },
        params_bin={
            'p_unmask_C': 0.9, 
            'p_unmask_T': 0.5,
        },
        confs = [7],
        )
    df_trialsData.append(df_trialData)

    ### Concatanate and save 
    df_trialsData = pd.concat(df_trialsData, ignore_index=True)
    df_trialsData.to_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv'), index=False)

if True: ### Calculate CIs according to various methods - can take a while to run

    df_trialsResults = power.Stats.get_df_trialsResults(
        df_trialsData = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv')),
        #sample_sizes = [160, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 400],
        sample_sizes = [20, 40, 60, 80, 100],
        rope_cgr=0.14, 
        rope_bbi=0.2, 
        rope_gmg=5,
        rope_gmgc=5,)        
    df_trialsResults.to_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv'), index=False)

if True: ### Calc averages across scenarios / sample sizes 

    df_power = power.Power.get_df_power(
        df_trialsResults = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv')),)
    df_power.to_csv(os.path.join(folders.power, f'{prefix}_power.csv'), index=False)

if False: ### Generate mock data

    prefix='tmp'
    scenario = 'tmp'

    n_trials = 1
    sample_size = 10000

    df_trialsData_exp = power.DataGeneration.get_df_trialsData_params(
        scenario = scenario,
        n_trials = n_trials, 
        sample_size = sample_size, 
        params_cont={
            'mean_C': 10.5, 
            'mean_T': 10.5,        
            'sd': 7.6, },
        params_bin={
            'p_unmask_C': 0.5, 
            'p_unmask_T': 0.9,
        },
        confs = [7],)

    confusion = power.Stats.get_confusion(df_trialsData_exp)
    df_trialsData_gen = power.DataGeneration.get_df_trialsData_confusion(
        scenario = scenario,
        confusion = confusion,
        n_trials=1, 
        sample_size=1000)

    samples_sizes = [sample_size for sample_size in range(10,210,10)]
    df_trialsResults = power.Stats.get_df_cis(df_trialsData_gen, methods=['cgr'], sample_sizes=samples_sizes)

    sampel_sizes = range(1,10, 160)
    for sample in samle_sizes:
        df_trialData
        cgr, cgr_ciL, cgr_ciH = power.Stats.calc_cgr_cis(df_trialData)
        
        cgr_ciL
