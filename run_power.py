import src.config as config
import src.power_calc as power
import src.folders as folders
import pandas as pd
import os


''' Trad power calc '''
prefix='tmp'

if False: ### Generate mock data

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

if False: ### Calculate CIs - can take a while to run

    df_CIs = power.Stats.get_df_CIs(
        df_trialsData = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv')),
        #sample_sizes = [160, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 400],
        sample_sizes = [20, 40, 60, 80, 100],)        
    
    df_trialsResults = power.Stats.get_df_trialsResults(
        df_CIs = df_CIs,)

    df_trialsResults.to_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv'), index=False)

if False: ### Calc averages across scenarios / sample sizes 

    df_power = power.Power.get_df_power(
        df_trialsResults = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv')),)
    df_power.to_csv(os.path.join(folders.power, f'{prefix}_power.csv'), index=False)


''' NNUM calc '''
prefix='nnum'

if False: ### Generate mock data

    n_trials = 30
    sample_size = 300
    df_trialsData=[]

    ### CGR 0.50
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr050', 
        confusion = {
            'trtC_guessC': 0.25,
            'trtC_guessT': 0.25,
            'trtT_guessC': 0.25,
            'trtT_guessT': 0.25,}, 
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    # ### CGR 0.55
    # df_trialData = power.DataGeneration.get_df_trialsData_confusion(
    #     scenario = 'cgr055', 
    #     confusion = {
    #         'trtC_guessC': 0.275,
    #         'trtC_guessT': 0.225,
    #         'trtT_guessC': 0.225,
    #         'trtT_guessT': 0.275,},
    #     n_trials = n_trials, 
    #     sample_size = sample_size)
    # df_trialsData.append(df_trialData)

    ### CGR 0.60
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr060', 
        confusion = {
            'trtC_guessC': 0.3,
            'trtC_guessT': 0.2,
            'trtT_guessC': 0.2,
            'trtT_guessT': 0.3,}, 
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    ### CGR 0.9
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr090', 
        confusion = {
            'trtC_guessC': 0.450,
            'trtC_guessT': 0.050,
            'trtT_guessC': 0.050,
            'trtT_guessT': 0.450,},
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    ### Concatanate and save 
    df_trialsData = pd.concat(df_trialsData, ignore_index=True)
    df_trialsData.to_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv'), index=False)

if False: ### Calculate CIs - can take a while to run

    df_CIs = power.Stats.get_df_CIs(
        df_trialsData = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv')),
        sample_sizes = [100, 140,180, 220, 260, 300],)        
    df_CIs.to_csv(os.path.join(folders.power, f'{prefix}_CIs.csv'), index=False)

    df_trialsResults = power.Stats.get_df_trialsResults(df_CIs = df_CIs, trim_CIs=False)
    df_trialsResults.to_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv'), index=False)

    df_power = power.Power.get_df_power(
        df_trialsResults = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv')),)
    df_power.to_csv(os.path.join(folders.power, f'{prefix}_power.csv'), index=False)

if True: ### Time comparison to Calculate CIs
    import time

    sample_sizes = [sample for sample in range(10, 310, 10)] #[100, 140,180, 220, 260, 300]

    start = time.time()
    df_CIs = power.Stats.get_df_CIs(
        df_trialsData = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv')),
        sample_sizes = sample_sizes,)
    print(f'Sequential: {time.time() - start:.2f}s')        
    df_CIs.to_csv(os.path.join(folders.power, f'{prefix}_sequential_CIs.csv'), index=False)

    start = time.time()
    df_CIs = power.Stats.get_df_CIs_parallel(
        df_trialsData = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv')),
        sample_sizes = sample_sizes,)
    print(f'Parallel: {time.time() - start:.2f}s')        
    df_CIs.to_csv(os.path.join(folders.power, f'{prefix}_parallel_CIs.csv'), index=False)


if False: 

    df_nnumBounds = power.Power.get_df_nnumBounds(
       df_trialsResults = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv')),)
    df_nnumBounds.to_csv(os.path.join(folders.power, f'{prefix}_nnumBounds.csv'), index=False)
