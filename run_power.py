import src.config as config
import src.power_calc as power
import src.folders as folders
import src.core as core
import pandas as pd
import itertools
import os


if True: ### Generate mock data

    n_trials = 150
    sample_size = 500
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

    # Optimistic scenario: mean dose guesses are 2mg apart, SD/confidence both moderate
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
    df_trials.to_csv(os.path.join(folders.power, f'adapt_trials.csv'), index=False)

    # # POP scenario: according to pop data; NO NEED TO SIMULATE IT, THERE IS NO CHANCE WE CAN SHOW EQUIVALENCE
    # adapt_trials_pop = power.DataGeneration.get_df_trials(
    #     n_trials = n_trials, 
    #     n_per_arm = n_per_arm, 
    #     mean_C = 4.5, 
    #     mean_T = 18,        
    #     sd = 11,
    #     confs = config.confs,
    #     )
    # adapt_trials_pop.to_csv(os.path.join(folders.power, 'adapt_trials_pop.csv'), index=False)

if True: ### Calculate CIs according to various methods - can take a while to run

    df_stats = power.Power.get_df_stats(
        df_trials = pd.read_csv(os.path.join(folders.power, 'adapt_trials.csv')),
        sample_sizes = [80, 100, 120, 140, 160, 180, 200, 300, 400, 500],
        rope_cgr=0.14, 
        rope_bbi=0.2, 
        rope_mix=5,)        
    df_stats.to_csv(os.path.join(folders.power, 'adapt_stats.csv'), index=False)

if True: ### Calc averages across scenarios / sample sizes 

    df_power = power.Power.get_df_power(
        df_stats = pd.read_csv(os.path.join(folders.power, 'adapt_stats.csv')),)
    df_power.to_csv(os.path.join(folders.power, f'adapt_power.csv'), index=False)