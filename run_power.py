import src.config as config
import src.power_calc as power
import src.folders as folders
import pandas as pd
import time
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

if True: ### Generate mock data

    n_trials = 500
    sample_size = 300
    df_trialsData=[]

    ### CGR 0.50
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr_0.500', 
        confusion = {
            'trtC_guessC': 0.25,
            'trtC_guessT': 0.25,
            'trtT_guessC': 0.25,
            'trtT_guessT': 0.25,}, 
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    ### CGR 0.525
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr_0.525', 
        confusion = {
            'trtC_guessC': 0.2625,
            'trtC_guessT': 0.2375,
            'trtT_guessC': 0.2375,
            'trtT_guessT': 0.2625,},
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    ### CGR 0.55
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr_0.550', 
        confusion = {
            'trtC_guessC': 0.275,
            'trtC_guessT': 0.225,
            'trtT_guessC': 0.225,
            'trtT_guessT': 0.275,},
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    ### CGR 0.575
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr_0.575', 
        confusion = {
            'trtC_guessC': 0.2875,
            'trtC_guessT': 0.2125,
            'trtT_guessC': 0.2125,
            'trtT_guessT': 0.2875,},
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    ### CGR 0.600
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr_0.600', 
        confusion = {
            'trtC_guessC': 0.3,
            'trtC_guessT': 0.2,
            'trtT_guessC': 0.2,
            'trtT_guessT': 0.3,}, 
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    ### CGR 0.625
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr_0.625', 
        confusion = {
            'trtC_guessC': 0.3125,
            'trtC_guessT': 0.1875,
            'trtT_guessC': 0.1875,
            'trtT_guessT': 0.3125,}, 
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    ### CGR 0.650
    df_trialData = power.DataGeneration.get_df_trialsData_confusion(
        scenario = 'cgr_0.650', 
        confusion = {
            'trtC_guessC': 0.325,
            'trtC_guessT': 0.175,
            'trtT_guessC': 0.175,
            'trtT_guessT': 0.325,}, 
        n_trials = n_trials, 
        sample_size = sample_size)
    df_trialsData.append(df_trialData)

    # ### CGR 0.675
    # df_trialData = power.DataGeneration.get_df_trialsData_confusion(
    #     scenario = 'cgr_0.675', 
    #     confusion = {
    #         'trtC_guessC': 0.3375,
    #         'trtC_guessT': 0.1625,
    #         'trtT_guessC': 0.1625,
    #         'trtT_guessT': 0.3375,}, 
    #     n_trials = n_trials, 
    #     sample_size = sample_size)
    # df_trialsData.append(df_trialData)

    # ### CGR 0.700
    # df_trialData = power.DataGeneration.get_df_trialsData_confusion(
    #     scenario = 'cgr_0.700', 
    #     confusion = {
    #         'trtC_guessC': 0.35,
    #         'trtC_guessT': 0.15,
    #         'trtT_guessC': 0.15,
    #         'trtT_guessT': 0.35,}, 
    #     n_trials = n_trials, 
    #     sample_size = sample_size)
    # df_trialsData.append(df_trialData)

    # ### CGR 0.750
    # df_trialData = power.DataGeneration.get_df_trialsData_confusion(
    #     scenario = 'cgr_0.750', 
    #     confusion = {
    #         'trtC_guessC': 0.375,
    #         'trtC_guessT': 0.125,
    #         'trtT_guessC': 0.125,
    #         'trtT_guessT': 0.375,}, 
    #     n_trials = n_trials, 
    #     sample_size = sample_size)
    # df_trialsData.append(df_trialData)

    # ### CGR 0.95
    # df_trialData = power.DataGeneration.get_df_trialsData_confusion(
    #     scenario = 'cgr095', 
    #     confusion = {
    #         'trtC_guessC': 0.475,
    #         'trtC_guessT': 0.025,
    #         'trtT_guessC': 0.025,
    #         'trtT_guessT': 0.475,},
    #     n_trials = n_trials, 
    #     sample_size = sample_size)
    # df_trialsData.append(df_trialData)

    ### Concatanate and save 
    df_trialsData = pd.concat(df_trialsData, ignore_index=True)
    df_trialsData.to_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv'), index=False)

if True: ### Calculate CIs - can take a while to run

    sample_sizes = [sample_size for sample_size in range(20, 305, 10,)]

    start = time.time()
    df_CIs = power.Stats.get_df_CIs_vectorized(
        df_trialsData = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsData.csv')),
        sample_sizes = sample_sizes,)
    print(f'Calculating CIs took {time.time() - start:.2f}s')  

    df_trialsResults = power.Stats.get_df_trialsResults(
        df_CIs = df_CIs, 
        trim_CIs=False)
    df_trialsResults = power.Helpers.convert_res_to_numeric(df_trialsResults)

    df_trialsResults.to_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv'), index=False)

if False: 

    df_nnumBounds = power.Power.get_df_nnumBounds(
       df_trialsResults = pd.read_csv(os.path.join(folders.power, f'{prefix}_trialsResults.csv')),)
    df_nnumBounds.to_csv(os.path.join(folders.power, f'{prefix}_nnumBounds.csv'), index=False)
