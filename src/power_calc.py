from statsmodels.stats.proportion import proportion_confint
from scipy.stats import beta as beta_dist
from joblib import Parallel, delayed
from io import StringIO
from itertools import product
from tqdm import tqdm
import src.config as config
import pandas as pd
import numpy as np
import math

if 'bbi' in config.methods:
    from rpy2.robjects import r
    r('library(BI)')


class DataGeneration():

    @staticmethod
    def get_df_trialsData_params(scenario, n_trials, sample_size, params_bin, params_cont, thr_dose=config.thr_dose, confs=config.confs, conf_to_se=config.conf_to_se,):

        assert sample_size % 2==0, 'Sample_size must be even'
        n_per_arm = int(sample_size/2)

        ### Parameters for binary guess generation
        p_unmask_C = params_bin['p_unmask_C']
        p_unmask_T = params_bin['p_unmask_T']

        ### Parameters for continous guess generation
        mean_C = params_cont['mean_C']
        mean_T = params_cont['mean_T']
        sd = params_cont['sd']

        ### Generate data per trial / arm
        df_trialsData = pd.DataFrame()
        for trial in tqdm(range(0, n_trials), desc=f'Generate {scenario} data'):            
            df_trialData = pd.DataFrame()

            for arm in ['C', 'T']:
                
                if arm == 'C':
                    trts = ['C']*n_per_arm
                    guess_bins = np.random.choice(['C', 'T'], size=n_per_arm, p=[p_unmask_C, 1-p_unmask_C])
                    doses = np.random.normal(mean_C, sd, n_per_arm)
                elif arm == 'T':
                    trts = ['T']*n_per_arm
                    guess_bins = np.random.choice(['T', 'C'], size=n_per_arm, p=[p_unmask_T, 1-p_unmask_T])
                    doses = np.random.normal(mean_T, sd, n_per_arm)
                else:
                    assert False
                
                df_arm = pd.DataFrame({
                    'scenario': scenario,
                    'trial': [trial]*n_per_arm,
                    'trt': trts,
                    'guess_bin': guess_bins,
                    'guess_dose': [round(dose, 1) for dose in doses],
                    'guess_conf': np.random.choice(confs, n_per_arm),})

                df_trialData = pd.concat([df_trialData,  df_arm], ignore_index=True)

            ### Within each trial, randomize patient order, add patient ID, concatenate
            df_trialData = df_trialData.sample(frac=1).reset_index(drop=True)
            df_trialData['pID'] = df_trialData.index
            df_trialsData = pd.concat([df_trialData, df_trialsData], ignore_index=True)

        ### Generate guess SD and binary guess
        df_trialsData['guess_se'] = df_trialsData['guess_conf'].map(conf_to_se)
        df_trialsData['guess_bin_pop'] = df_trialsData['guess_dose'].apply(lambda x: 'T' if x >= thr_dose else 'C')
        
        df_trialsData = df_trialsData[['scenario', 'trial', 'pID', 'trt', 'guess_bin', 'guess_dose', 'guess_conf', 'guess_bin_pop', 'guess_se']]
        return df_trialsData

    @staticmethod
    def get_df_trialsData_confusion(scenario, confusion, n_trials=100, sample_size=1000):        

        ### Define probability of trt-guess pairs from confusion matrix
        trt_guess = ['CC', 'CT', 'TC', 'TT']
        probs = [
            confusion['trtC_guessC'], 
            confusion['trtC_guessT'], 
            confusion['trtT_guessC'], 
            confusion['trtT_guessT'], ]

        df_trialsData = pd.DataFrame()
        for trial in tqdm(range(0, n_trials), desc=f'Generate {scenario} data'):
            df_trialData = pd.DataFrame()

            ### Generate data
            results = np.random.choice(trt_guess, size=sample_size, p=probs)
            trts = [r[0] for r in results]
            guess_bins = [r[1] for r in results]
                    
            df_trialData = pd.DataFrame({
                'scenario': scenario,
                'trial': [trial]*sample_size,
                'trt': trts,
                'guess_bin': guess_bins,})

            ### Within each trial, randomize patient order, add patient ID, concatenate
            df_trialData['pID'] = df_trialData.index
            df_trialsData = pd.concat([df_trialData, df_trialsData], ignore_index=True)
        
        df_trialsData = df_trialsData[['scenario', 'trial', 'pID', 'trt', 'guess_bin']]
        return df_trialsData


class Stats():

    @staticmethod
    def get_df_CIs(df_trialsData, sample_sizes, methods=config.methods,):

        dfs = []
        scenarios = df_trialsData.scenario.unique()
        trials = df_trialsData.trial.unique()

        for scenario, trial, sample_size in tqdm(product(scenarios, trials, sample_sizes), desc='Calc CIs'):

            df_C = df_trialsData.loc[
                (df_trialsData.scenario==scenario) & 
                (df_trialsData.trial==trial) & 
                (df_trialsData.trt=='C')].reset_index().iloc[0:round(sample_size/2), :]
            df_T = df_trialsData.loc[
                (df_trialsData.scenario==scenario) & 
                (df_trialsData.trial==trial) & 
                (df_trialsData.trt=='T')].reset_index().iloc[0:round(sample_size/2), :]
            
            df_trialData = pd.concat([df_C, df_T], ignore_index=True).reset_index(drop=True)

            df_CI = Stats.calc_cis(
                df_trialData = df_trialData,
                methods = methods,)

            ### Add bookkeeping vars and append
            df_CI.insert(0, 'sample_size', sample_size) 
            df_CI.insert(0, 'trial', trial) 
            df_CI.insert(0, 'scenario', scenario)             
            dfs.append(df_CI)

        ### Concat CIs
        df_CIs = pd.concat(dfs, ignore_index=True)
        df_CIs = df_CIs.sort_values(by=['scenario', 'trial', 'sample_size',], ascending=True, ignore_index=True)
        return df_CIs

    @staticmethod ### Only works for CGR
    def get_df_CIs_vectorized(df_trialsData, sample_sizes, digits=config.digits):
        """Vectorized CGR CI calculation - much faster than loop-based approaches"""
        
        ### Add match column (trt == guess_bin)
        df = df_trialsData.copy()
        df['match'] = (df['trt'] == df['guess_bin']).astype(int)
        
        ### Add row number within each (scenario, trial, trt) group for sampling
        df['row_num'] = df.groupby(['scenario', 'trial', 'trt']).cumcount()
        
        rows = []
        for sample_size in sample_sizes:
            n_per_arm = round(sample_size / 2)
            
            ### Filter to first n_per_arm rows per (scenario, trial, trt)
            df_sample = df[df['row_num'] < n_per_arm]
            
            ### Group by (scenario, trial) and compute k, n
            grouped = df_sample.groupby(['scenario', 'trial']).agg(
                k=('match', 'sum'),
                n=('match', 'count')
            ).reset_index()
            
            grouped['sample_size'] = sample_size
            rows.append(grouped)
        
        ### Combine all sample sizes
        df_results = pd.concat(rows, ignore_index=True)
        
        ### Vectorized CI calculation using beta distribution
        k = df_results['k'].values
        n = df_results['n'].values
        alpha = 0.05
        
        df_results['cgr'] = (k / n).round(digits)
        df_results['cgr_ciL'] = beta_dist.ppf(alpha / 2, k, n - k + 1).round(digits)
        df_results['cgr_ciH'] = beta_dist.ppf(1 - alpha / 2, k + 1, n - k).round(digits)
        df_results['cgr_moe'] = ((df_results['cgr_ciH'] - df_results['cgr_ciL']) / 2).round(digits)
        
        ### Handle edge cases (k=0 or k=n)
        df_results.loc[k==0, 'cgr_ciL'] = 0.0
        df_results.loc[k==n, 'cgr_ciH'] = 1.0
        
        ### Select and order columns
        df_CIs = df_results[['scenario', 'trial', 'sample_size', 'cgr', 'cgr_ciL', 'cgr_ciH', 'cgr_moe']]    
        df_CIs = df_CIs.sort_values(by=['scenario', 'trial', 'sample_size',], ascending=True, ignore_index=True)
        return df_CIs
        
    @staticmethod
    def get_df_trialsResults(df_CIs, trim_CIs=True):

        df_trialsResults = df_CIs.copy()

        ### Add trial results 
        df_trialsResults = Stats.add_eqv(
            df_CIs = df_trialsResults,)

        df_trialsResults = Stats.add_nsd(
            df_CIs = df_trialsResults,)

        df_trialsResults = Stats.add_sd(
            df_CIs = df_trialsResults,)

        if trim_CIs:
            rm_cols = [col for col in df_trialsResults.columns if (('_ciL' in col) | ('_ciH' in col))]
            df_trialsResults = df_trialsResults.drop(columns=rm_cols)
            
        return df_trialsResults

    @staticmethod
    def add_sd(df_CIs):

        if all([col in df_CIs.columns for col in ['cgr_ciL', 'cgr_ciH']]):            
            pos = df_CIs.columns.get_loc('cgr_moe') + 1
            df_CIs.insert(pos, 'cgr_sd', None) 
            df_CIs['cgr_sd'] = (
                (df_CIs['cgr_ciL'] >= 0.5) | (df_CIs['cgr_ciH'] <= 0.5))

        if all([col in df_CIs.columns for col in ['bbi_C_ciL', 'bbi_C_ciH']]):
            pos = df_CIs.columns.get_loc('bbi_C_moe') + 1
            df_CIs.insert(pos, 'bbi_C_sd', None) 
            df_CIs['bbi_C_sd'] = (
                (df_CIs['bbi_C_ciL'] >= 0) | (df_CIs['bbi_C_ciH'] <= 0))

        if all([col in df_CIs.columns for col in ['bbi_T_ciL', 'bbi_T_ciH']]):
            pos = df_CIs.columns.get_loc('bbi_T_moe') + 1
            df_CIs.insert(pos, 'bbi_T_sd', None) 
            df_CIs['bbi_T_sd'] = (
                (df_CIs['bbi_T_ciL'] >= 0) | (df_CIs['bbi_T_ciH'] <= 0))

        if all([col in df_CIs.columns for col in ['gmg_ciL', 'gmg_ciH']]):            
            pos = df_CIs.columns.get_loc('gmg_moe') + 1
            df_CIs.insert(pos, 'gmg_sd', None) 
            df_CIs['gmg_sd'] = (
                (df_CIs['gmg_ciL'] >= 0) | (df_CIs['gmg_ciH'] <= 0))    

        if all([col in df_CIs.columns for col in ['gmgc_ciL', 'gmgc_ciH']]):            
            pos = df_CIs.columns.get_loc('gmgc_moe') + 1
            df_CIs.insert(pos, 'gmgc_sd', None) 
            df_CIs['gmgc_sd'] = (
                (df_CIs['gmgc_ciL'] >= 0) | (df_CIs['gmgc_ciH'] <= 0))

        return df_CIs      
        
    @staticmethod
    def add_nsd(df_CIs):

        if all([col in df_CIs.columns for col in ['cgr_ciL', 'cgr_ciH']]):            
            pos = df_CIs.columns.get_loc('cgr_moe') + 1
            df_CIs.insert(pos, 'cgr_nsd', None) 
            df_CIs['cgr_nsd'] = (
                (df_CIs['cgr_ciL'] < 0.5) &
                (df_CIs['cgr_ciH'] > 0.5))

        if all([col in df_CIs.columns for col in ['bbi_C_ciL', 'bbi_C_ciH']]):
            pos = df_CIs.columns.get_loc('bbi_C_moe') + 1
            df_CIs.insert(pos, 'bbi_C_nsd', None) 
            df_CIs['bbi_C_nsd'] = (
                (df_CIs['bbi_C_ciL'] < 0) &
                (df_CIs['bbi_C_ciH'] > 0))

        if all([col in df_CIs.columns for col in ['bbi_T_ciL', 'bbi_T_ciH']]):
            pos = df_CIs.columns.get_loc('bbi_T_moe') + 1
            df_CIs.insert(pos, 'bbi_T_nsd', None) 
            df_CIs['bbi_T_nsd'] = (
                (df_CIs['bbi_T_ciL'] < 0) &
                (df_CIs['bbi_T_ciH'] > 0))

        if all([col in df_CIs.columns for col in ['gmg_ciL', 'gmg_ciH']]):            
            pos = df_CIs.columns.get_loc('gmg_moe') + 1
            df_CIs.insert(pos, 'gmg_nsd', None) 
            df_CIs['gmg_nsd'] = (
                (df_CIs['gmg_ciL'] < 0) &
                (df_CIs['gmg_ciH'] > 0))      

        if all([col in df_CIs.columns for col in ['gmgc_ciL', 'gmgc_ciH']]):            
            pos = df_CIs.columns.get_loc('gmgc_moe') + 1
            df_CIs.insert(pos, 'gmgc_nsd', None) 
            df_CIs['gmgc_nsd'] = (
                (df_CIs['gmgc_ciL'] < 0) &
                (df_CIs['gmgc_ciH'] > 0))    

        return df_CIs        

    @staticmethod
    def add_eqv(df_CIs, ropes=config.ropes):
        
        if all([col in df_CIs.columns for col in ['cgr_ciL', 'cgr_ciH']]):    
            rope_cgr = ropes['cgr']        
            pos = df_CIs.columns.get_loc('cgr_moe') + 1
            df_CIs.insert(pos, 'cgr_eqv', None) 
            df_CIs['cgr_eqv'] = (
                (df_CIs['cgr_ciH'] < 0.5 + rope_cgr) &
                (df_CIs['cgr_ciL']  > 0.5 - rope_cgr))           

        if all([col in df_CIs.columns for col in ['bbi_C_ciL', 'bbi_C_ciH']]):
            rope_bbi = ropes['bbi']
            pos = df_CIs.columns.get_loc('bbi_C_moe') + 1
            df_CIs.insert(pos, 'bbi_C_eqv', None) 
            df_CIs['bbi_C_eqv'] = (
                (df_CIs['bbi_C_ciH'] < rope_bbi) &
                (df_CIs['bbi_C_ciL']  > -rope_bbi))

        if all([col in df_CIs.columns for col in ['bbi_T_ciL', 'bbi_T_ciH']]):
            rope_bbi = ropes['bbi']
            pos = df_CIs.columns.get_loc('bbi_T_moe') + 1
            df_CIs.insert(pos, 'bbi_T_eqv', None) 
            df_CIs['bbi_T_eqv'] = (
                (df_CIs['bbi_T_ciH'] < rope_bbi) &
                (df_CIs['bbi_T_ciL']  > -rope_bbi))

        if all([col in df_CIs.columns for col in ['gmg_ciL', 'gmg_ciH']]):       
            rope_gmg = ropes['gmg']                         
            pos = df_CIs.columns.get_loc('gmg_moe') + 1
            df_CIs.insert(pos, 'gmg_eqv', None) 
            df_CIs['gmg_eqv'] = (
                (df_CIs['gmg_ciH'] < rope_gmg) &
                (df_CIs['gmg_ciL']  > -rope_gmg))      

        if all([col in df_CIs.columns for col in ['gmgc_ciL', 'gmgc_ciH']]):    
            rope_gmgc = ropes['gmgc']        
            pos = df_CIs.columns.get_loc('gmgc_moe') + 1
            df_CIs.insert(pos, 'gmgc_eqv', None) 
            df_CIs['gmgc_eqv'] = (
                (df_CIs['gmgc_ciH'] < rope_gmgc) &
                (df_CIs['gmgc_ciL']  > -rope_gmgc))    

        return df_CIs
    
    @staticmethod
    def get_confusion(df_trialsData, col_guess='guess_bin', digits=4):
        
        total_sample = df_trialsData.shape[0]
        confusion = {
            'trtC_guessC': round(df_trialsData.loc[(df_trialsData.trt=='C') & (df_trialsData[f'{col_guess}']=='C')].shape[0]/total_sample, digits),
            'trtT_guessC': round(df_trialsData.loc[(df_trialsData.trt=='T') & (df_trialsData[f'{col_guess}']=='C')].shape[0]/total_sample, digits),
            'trtC_guessT': round(df_trialsData.loc[(df_trialsData.trt=='C') & (df_trialsData[f'{col_guess}']=='T')].shape[0]/total_sample, digits),
            'trtT_guessT': round(df_trialsData.loc[(df_trialsData.trt=='T') & (df_trialsData[f'{col_guess}']=='T')].shape[0]/total_sample, digits),
        }

        return confusion


    ''' Calculate CIs '''
    @staticmethod
    def calc_cis(df_trialData, methods=config.methods, digits=config.digits):
        ''' Calculate CI of various blinding metrics for various sample sizes '''

        assert len(df_trialData.scenario.unique())==1, 'Calculating CI across mulitple scenarios'
        assert len(df_trialData.trial.unique())==1, 'Calculating trial across mulitple scenarios'        

        row={}
        ### Get CGR stats
        if 'cgr' in methods:
            cgr, cgr_ciL, cgr_ciH = Stats.calc_cgr_cis(df_trialData)
            row['cgr'] = cgr
            row['cgr_ciL'] = cgr_ciL
            row['cgr_ciH'] = cgr_ciH
            row['cgr_moe'] = (cgr_ciH-cgr_ciL)/2
        if 'bbi' in methods:
            bbi_C, bbi_C_ciL, bbi_C_ciH, bbi_T, bbi_T_ciL, bbi_T_ciH = Stats.calc_bbi_cis(df_trialData)
            row['bbi_C']= bbi_C
            row['bbi_C_ciL']= bbi_C_ciL
            row['bbi_C_ciH']= bbi_C_ciH
            row['bbi_C_moe'] = (bbi_C_ciH-bbi_C_ciL)/2
            row['bbi_T']= bbi_T
            row['bbi_T_ciL']= bbi_T_ciL
            row['bbi_T_ciH']= bbi_T_ciH
            row['bbi_T_moe'] = (bbi_T_ciH-bbi_T_ciL)/2
        if 'gmg' in methods:
            gmg, gmg_ciL, gmg_ciH = Stats.calc_gmg_cis(df_trialData)
            row['gmg']= gmg
            row['gmg_ciL']= gmg_ciL
            row['gmg_ciH']= gmg_ciH
            row['gmg_moe'] = (gmg_ciH-gmg_ciL)/2        
        if 'gmgc' in methods:
            gmgc, gmgc_ciL, gmgc_ciH = Stats.calc_gmgc_cis(df_trialData)
            row['gmgc']= gmgc
            row['gmgc_ciL']= gmgc_ciL
            row['gmgc_ciH']= gmgc_ciH
            row['gmgc_moe'] = (gmgc_ciH-gmgc_ciL)/2

        ### Convert to DF
        df_ci = pd.DataFrame([row])

        ### Round columns
        cols_to_round = df_ci.columns.tolist()
        df_ci[cols_to_round] = df_ci[cols_to_round].round(digits)        
        
        return df_ci

    @staticmethod ###  NEED TO GENERALIZE TO guess_bin_pop / guess_bin
    def calc_cgr_cis(df_trialData):
            
        matches = (df_trialData['trt'] == df_trialData['guess_bin']).astype(int)
        k = matches.sum()
        n = len(matches)
        cgr = k / n
        cgr_ciL, cgr_ciH = proportion_confint(k, n, alpha=0.05, method="beta")

        return cgr, cgr_ciL, cgr_ciH

    @staticmethod
    def calc_bbi_cis(df_trialData):
        
        n_CC = df_trialData.loc[(df_trialData.trt=='C') & (df_trialData.guess_bin_pop=='C')].shape[0]
        n_CT = df_trialData.loc[(df_trialData.trt=='C') & (df_trialData.guess_bin_pop=='T')].shape[0]
        n_TC = df_trialData.loc[(df_trialData.trt=='T') & (df_trialData.guess_bin_pop=='C')].shape[0]
        n_TT = df_trialData.loc[(df_trialData.trt=='T') & (df_trialData.guess_bin_pop=='T')].shape[0]

        ### Calculate BBI in R and then convert results to pandas df
        r(f'BI = BI(matrix(c({n_TT}, {n_CT}, {n_TC}, {n_CC}, 0, 0), nrow = 3, ncol = 2, byrow = TRUE))')    
        df_bbi = Helpers.rBI2df(str(r('BI$BangBI')))

        bbi_C = df_bbi.loc[(df_bbi.assigned=='Placebo')].reset_index().est[0]
        bbi_C_ciL = df_bbi.loc[(df_bbi.assigned=='Placebo')].reset_index().bbi_ciL[0]
        bbi_C_ciH = df_bbi.loc[(df_bbi.assigned=='Placebo')].reset_index().bbi_ciH[0]
        bbi_T = df_bbi.loc[(df_bbi.assigned=='Treatment')].reset_index().est[0]
        bbi_T_ciL = df_bbi.loc[(df_bbi.assigned=='Treatment')].reset_index().bbi_ciL[0]
        bbi_T_ciH =df_bbi.loc[(df_bbi.assigned=='Treatment')].reset_index().bbi_ciH[0]

        return bbi_C, bbi_C_ciL, bbi_C_ciH, bbi_T, bbi_T_ciL, bbi_T_ciH 

    @staticmethod
    def calc_gmg_cis(df_trialData):

        ### Calculate mean/SE of the dose estimate in each arm
        guess_mg_C = df_trialData.loc[(df_trialData.trt=='C')].guess_dose
        mean_C = guess_mg_C.mean()
        se_C = (guess_mg_C.std() / np.sqrt(len(guess_mg_C)))

        guess_mg_T = df_trialData.loc[(df_trialData.trt=='T')].guess_dose
        mean_T = guess_mg_T.mean()
        se_T = (guess_mg_T.std() / np.sqrt(len(guess_mg_T)))

        ### Calculate the SE of the difference
        se_diff = np.sqrt(se_C**2 + se_T**2)
        
        gmg = mean_T - mean_C
        gmg_ciL = gmg - (1.96*se_diff)
        gmg_ciH = gmg + (1.96*se_diff)
    
        return gmg, gmg_ciL, gmg_ciH

    @staticmethod
    def calc_gmgc_cis(df_trialData):       

        ### Calculate mean/SE of the dose estimate in each arm
        guess_mg_C = np.array(df_trialData.loc[(df_trialData.trt=='C')].guess_dose)
        guess_mg_se_C = np.array(df_trialData.loc[(df_trialData.trt=='C')].guess_se)        
        mean_C = np.mean(guess_mg_C) 
        se_C = np.sqrt(np.sum(guess_mg_se_C**2))/len(guess_mg_se_C)

        guess_mg_T = np.array(df_trialData.loc[(df_trialData.trt=='T')].guess_dose)
        guess_mg_se_T = np.array(df_trialData.loc[(df_trialData.trt=='T')].guess_se)        
        mean_T = np.mean(guess_mg_T) 
        se_T = np.sqrt(np.sum(guess_mg_se_T**2))/len(guess_mg_se_T)

        ### Calculate the SE of the difference 
        se_diff = np.sqrt(se_C**2+se_T**2)
        
        gmgc = mean_T - mean_C
        gmgc_ciL = gmgc - (1.96*se_diff)
        gmgc_ciH = gmgc + (1.96*se_diff)

        return gmgc, gmgc_ciL, gmgc_ciH


class Power():

    @staticmethod
    def get_df_power(df_trialsResults, digits=config.digits, methods=config.methods):

        df_trialsResults = Helpers.convert_res_to_numeric(df_trialsResults)
        sample_sizes = df_trialsResults.sample_size.unique()
        scenarios = df_trialsResults.scenario.unique()
        rows = []        

        ### Calculate average across trials for each scenario / sample size
        for scenario, sample_size in tqdm(product(scenarios, sample_sizes), desc='Calc df_power'):            

            df_sample = df_trialsResults.loc[(df_trialsResults.scenario==scenario) & (df_trialsResults.sample_size==sample_size)]    
            if df_sample.shape[0]==0:
                continue

            row={}
            row['scenario'] = scenario
            row['sample_size'] = sample_size
    
            if 'cgr' in methods:
                row['cgr'] = df_sample.cgr.mean()
                row['cgr_ciL'] = df_sample.cgr_ciL.mean()
                row['cgr_ciH'] = df_sample.cgr_ciH.mean()
                row['cgr_moe'] = (df_sample.cgr_ciH.mean()-df_sample.cgr_ciL.mean())/2
                row['cgr_nsd'] = df_sample.cgr_nsd.mean()
                row['cgr_eqv'] = df_sample.cgr_eqv.mean()
            if 'bbi' in methods:
                row['bbi_C'] = df_sample.bbi_C.mean()
                row['bbi_C_ciL'] = df_sample.bbi_C_ciL.mean()
                row['bbi_C_ciH'] = df_sample.bbi_C_ciH.mean()
                row['bbi_C_moe'] = (df_sample.bbi_C_ciH.mean()-df_sample.bbi_C_ciL.mean())/2
                row['bbi_C_nsd'] = df_sample.bbi_C_nsd.mean()
                row['bbi_C_eqv'] = df_sample.bbi_C_eqv.mean()
                row['bbi_T'] = df_sample.bbi_T.mean()
                row['bbi_T_ciL'] = df_sample.bbi_T_ciL.mean()
                row['bbi_T_ciH'] = df_sample.bbi_T_ciH.mean()
                row['bbi_T_moe'] = (df_sample.bbi_T_ciH.mean()-df_sample.bbi_T_ciL.mean())/2
                row['bbi_T_nsd'] = df_sample.bbi_T_nsd.mean()
                row['bbi_T_eqv'] = df_sample.bbi_T_eqv.mean()
            if 'gmg' in methods:
                row['gmg'] = df_sample.gmg.mean()
                row['gmg_ciL'] = df_sample.gmg_ciL.mean()
                row['gmg_ciH'] = df_sample.gmg_ciH.mean()
                row['gmg_moe'] = (df_sample.gmg_ciH.mean()-df_sample.gmg_ciL.mean())/2
                row['gmg_nsd'] = df_sample.gmg_nsd.mean()
                row['gmg_eqv'] = df_sample.gmg_eqv.mean()
            if 'gmgc' in methods:
                row['gmgc'] = df_sample.gmgc.mean()
                row['gmgc_ciL'] = df_sample.gmgc_ciL.mean()
                row['gmgc_ciH'] = df_sample.gmgc_ciH.mean()
                row['gmgc_moe'] = (df_sample.gmgc_ciH.mean()-df_sample.gmgc_ciL.mean())/2
                row['gmgc_nsd'] = df_sample.gmgc_nsd.mean()                                
                row['gmgc_eqv'] = df_sample.gmgc_eqv.mean()       

            rows.append(row)                         
            
        ### Turn rows into df     
        df_power = pd.DataFrame(rows)

        ### Round columns
        cols_to_round = df_power.columns.tolist()
        cols_to_round.remove('scenario')
        cols_to_round.remove('sample_size')
        df_power[cols_to_round] = df_power[cols_to_round].round(digits)        

        return df_power

    @staticmethod
    def get_df_nnumBounds(df_trialsResults,):

        scenarios = df_trialsResults.scenario.unique()
        trials = df_trialsResults.trial.unique()
        rows=[]
        
        for scenario, trial in tqdm(product(scenarios, trials), desc='Calc nnum bounds'):
            
            df_trialResults = df_trialsResults.loc[(df_trialsResults.scenario==scenario) & (df_trialsResults.trial==trial)]

            ### Get lower bound of the nnum
            df_trialResults = df_trialResults.sort_values(by='sample_size', ascending=False).reset_index()
            df_sd_False = df_trialResults.loc[df_trialResults['cgr_sd']==False, 'sample_size']
            if df_sd_False.shape[0]==0:
                nnumL = 0
            else:
                nnumL = df_sd_False.iloc[0] 

            ### Get upper bound of the nnum
            df_trialResults = df_trialResults.sort_values(by='sample_size', ascending=True).reset_index()
            df_sd_True = df_trialResults.loc[df_trialResults['cgr_sd']==True, 'sample_size']
            if df_sd_True.shape[0]==0:
                nnumH = math.nan
            else:
                nnumH = df_sd_True.iloc[0] 

            rows.append({
                'scenario': scenario,
                'trial': trial, 
                'nnumL': nnumL,
                'nnumH': nnumH,})

        df_nnumBounds = pd.DataFrame(rows)
        return df_nnumBounds
    
    @staticmethod
    def get_df_nnum(df_trialsResults,):

        scenarios = df_trialsResults.scenario.unique()
        trials = df_trialsResults.trial.unique()
        sample_sizes = df_trialsResults.sample_size.unique()
        rows=[]
        
        for scenario, trial, sample_size in tqdm(product(scenarios, trials, sample_sizes), desc='Calc nnum'):
            
            df_trialResults = df_trialsResults.loc[
                (df_trialsResults.scenario==scenario) & 
                (df_trialsResults.trial==trial) & 
                (df_trialsResults.sample_size==sample_size)]

            ### Get lower bound of the nnum
            df_trialResults = df_trialResults.sort_values(by='sample_size', ascending=False).reset_index()
            df_sd_False = df_trialResults.loc[df_trialResults['cgr_sd']==False, 'sample_size']
            if df_sd_False.shape[0]==0:
                nnumL = 0
            else:
                nnumL = df_sd_False.iloc[0] 

            ### Get upper bound of the nnum
            df_trialResults = df_trialResults.sort_values(by='sample_size', ascending=True).reset_index()
            df_sd_True = df_trialResults.loc[df_trialResults['cgr_sd']==True, 'sample_size']
            if df_sd_True.shape[0]==0:
                nnumH = math.nan
            else:
                nnumH = df_sd_True.iloc[0] 

            rows.append({
                'scenario': scenario,
                'trial': trial, 
                'nnumL': nnumL,
                'nnumH': nnumH,})

        df_nnumBounds = pd.DataFrame(rows)
        return df_nnumBounds


class Helpers():

    @staticmethod
    def convert_res_to_numeric(df_trialsResults):

        ### Convert EQV/NSD results to numeric 
        if 'cgr_eqv' in df_trialsResults.columns:
            df_trialsResults['cgr_eqv'] = df_trialsResults['cgr_eqv'].astype(int)

        if 'bbi_C_eqv' in df_trialsResults.columns:
            df_trialsResults['bbi_C_eqv'] = df_trialsResults['bbi_C_eqv'].astype(int)

        if 'bbi_T_eqv' in df_trialsResults.columns:
            df_trialsResults['bbi_T_eqv'] = df_trialsResults['bbi_T_eqv'].astype(int)
    
        if 'gmg_eqv' in df_trialsResults.columns:
            df_trialsResults['gmg_eqv'] = df_trialsResults['gmg_eqv'].astype(int)

        if 'gmgc_eqv' in df_trialsResults.columns:
            df_trialsResults['gmgc_eqv'] = df_trialsResults['gmgc_eqv'].astype(int)

        if 'cgr_nsd' in df_trialsResults.columns:
            df_trialsResults['cgr_nsd'] = df_trialsResults['cgr_nsd'].astype(int)

        if 'bbi_C_nsd' in df_trialsResults.columns:
            df_trialsResults['bbi_C_nsd'] = df_trialsResults['bbi_C_nsd'].astype(int)

        if 'bbi_T_nsd' in df_trialsResults.columns:
            df_trialsResults['bbi_T_nsd'] = df_trialsResults['bbi_T_nsd'].astype(int)
        
        if 'gmg_nsd' in df_trialsResults.columns:
            df_trialsResults['gmg_nsd'] = df_trialsResults['gmg_nsd'].astype(int)

        if 'gmgc_nsd' in df_trialsResults.columns:
            df_trialsResults['gmgc_nsd'] = df_trialsResults['gmgc_nsd'].astype(int)

        return df_trialsResults

    @staticmethod
    def rBI2df(str_rBI): # Convert R stringvector to pandas DF
        # Strip outer quotes
        cleaned = str_rBI.strip().strip("'")

        # Skip the header line, parse data only
        lines = cleaned.strip().split('\n')
        data_lines = '\n'.join(lines[1:])  # skip header

        # Parse data rows (whitespace-separated, first value is row name)
        df = pd.read_csv(StringIO(data_lines), sep=r'\s+', header=None)

        # Assign correct column names
        df.columns = ['assigned', 'est', 'se', 'bbi_ciL', 'bbi_ciH']

        return df