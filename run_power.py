import src.config as config
import src.power_calc as power
import src.folders as folders
import src.core as core
import pandas as pd
import itertools
import os

prefix='adapt'

if True: ### Generate mock data

    n_trials = 200
    sample_size = 400
    dfs = []
    
    ### Ideal scenario: mean dose guesses are the same, SD/confidence is generously small/high 
    scenario = 'ideal'
    df_trial = power.DataGeneration.get_df_trials(
        scenario = scenario,
        n_trials = n_trials, 
        sample_size = sample_size, 
        mean_C = 10.5, 
        mean_T = 10.5,        
        sd = 11/3, # third of real SD
        confs = [5,6,6,7,7,7], # only have high conf guesses
        )
    dfs.append(df_trial)

    # Optimistic scenario with mean dose guesses are 2mg apart, SD/confidence both moderate
    scenario = 'diff2mg'
    df_trial = power.DataGeneration.get_df_trials(
        scenario = scenario,
        n_trials = n_trials, 
        sample_size = sample_size, 
        mean_C = 9.5, 
        mean_T = 11.5,        
        sd = 11/2, # half of real SD
        confs =  config.confs.tolist()+[7.0]*12, # Add more high confidence guesses
        )
    dfs.append(df_trial)

    # Optimistic scenario with mean dose guesses are 4mg apart, SD/confidence both moderate
    scenario = 'diff4mg'
    df_trial = power.DataGeneration.get_df_trials(
        scenario = scenario,
        n_trials = n_trials, 
        sample_size = sample_size, 
        mean_C = 8.5, 
        mean_T = 12.5,        
        sd = 11/2, # half of real SD
        confs = config.confs.tolist()+[7.0]*12, # Add more high confidence guesses
        )
    dfs.append(df_trial)

    ### Concatanate and save 
    df_trials = pd.concat(dfs, ignore_index=True)
    df_trials.to_csv(os.path.join(folders.power, f'{prefix}_trials.csv'), index=False)

if True: ### Calculate CIs according to various methods - can take a while to run

    df_stats = power.Stats.get_df_cis(
        df_trials = pd.read_csv(os.path.join(folders.power, f'{prefix}_trials.csv')),
        sample_sizes = [80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 400],
        #sample_sizes = [50],
        rope_cgr=0.14, 
        rope_bbi=0.2, 
        rope_gmg=5,
        rope_gmgc=5,)        
    df_stats.to_csv(os.path.join(folders.power, f'{prefix}_stats.csv'), index=False)

if True: ### Calc averages across scenarios / sample sizes 

    df_power = power.Power.get_df_power(
        df_cis = pd.read_csv(os.path.join(folders.power, f'{prefix}_cis.csv')),)
    df_power.to_csv(os.path.join(folders.power, f'{prefix}_power.csv'), index=False)