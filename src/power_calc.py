from statsmodels.stats.proportion import proportion_confint
from rpy2.robjects import r
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from tqdm import tqdm
import src.config as config
import pandas as pd
import numpy as np
import scipy

r('library(BI)')


class DataGeneration():

    @staticmethod ### Generate mock trial data 
    def get_df_trials(n_trials, sample_size, mean_C, mean_T, sd, confs, scenario='tmp'):

        df_trials = pd.DataFrame()
        assert sample_size % 2==0, "sample_size must be even"
        n_per_arm = int(sample_size/2)

        ### Generate data per trial / arm
        for trial in tqdm(range(0, n_trials), desc=f'Gen {scenario} trial data'):
            
            df_trial = pd.DataFrame()
            for arm in ['C', 'T']:
                
                if arm == 'C':
                    guesses = np.random.normal(mean_C, sd, n_per_arm)
                    trt = ['C']*n_per_arm
                elif arm == 'T':
                    guesses = np.random.normal(mean_T, sd, n_per_arm)
                    trt = ['T']*n_per_arm
                else:
                    assert False
                
                df_arm = pd.DataFrame({
                    'scenario': scenario,
                    'trial': [trial]*n_per_arm,
                    'trt': trt,
                    'guess_dose': [round(guess, 1) for guess in guesses],
                    'guess_conf': np.random.choice(confs, n_per_arm),})

                df_trial = pd.concat([df_trial,  df_arm], ignore_index=True)

            ### Randomize patient order, add patient ID and convert guess confidence to SDs
            df_trial = df_trial.sample(frac=1).reset_index(drop=True)
            df_trial['pID'] = df_trial.index
            df_trials = pd.concat([df_trial, df_trials], ignore_index=True)

        ### Generate guess SD and binary guess
        df_trials['guess_sd'] = df_trials['guess_conf'].map(config.conf_to_sd)
        df_trials['guess_bin'] = df_trials['guess_dose'].apply(lambda x: 'T' if x >= config.thr_dose else 'C')
        
        df_trials = df_trials[['scenario', 'trial', 'pID', 'trt', 'guess_bin', 'guess_dose', 'guess_conf', 'guess_sd']]
        return df_trials


class Stats():
    
    @staticmethod
    def calc_cgr_cis(df_trial, digits=3):
        
        matches = (df_trial['trt'] == df_trial['guess_bin']).astype(int)
        k = matches.sum()
        n = len(matches)
        cgr = k / n
        cgr_ci_low, cgr_ci_high = proportion_confint(k, n, alpha=0.05, method="beta")

        return round(cgr, digits), round(cgr_ci_low, digits), round(cgr_ci_high, digits)

    @staticmethod
    def calc_bbi_cis(df_trial, digits=3):
        
        n_CC = df_trial.loc[(df_trial.trt=='C') & (df_trial.guess_bin=='C')].shape[0]
        n_CT = df_trial.loc[(df_trial.trt=='C') & (df_trial.guess_bin=='T')].shape[0]
        n_TC = df_trial.loc[(df_trial.trt=='T') & (df_trial.guess_bin=='C')].shape[0]
        n_TT = df_trial.loc[(df_trial.trt=='T') & (df_trial.guess_bin=='T')].shape[0]

        ### Calculate BBI in R and then convert results to pandas df
        r(f'BI = BI(matrix(c({n_TT}, {n_CT}, {n_TC}, {n_CC}, 0, 0), nrow = 3, ncol = 2, byrow = TRUE))')    
        df_bbi = Stats.rBI2df(str(r('BI$BangBI')))

        bbi_C = round(df_bbi.loc[(df_bbi.assigned=='Placebo')].reset_index().est[0], digits)
        bbi_C_ci_low = round(df_bbi.loc[(df_bbi.assigned=='Placebo')].reset_index().bbi_ci_low[0], digits)
        bbi_C_ci_high = round(df_bbi.loc[(df_bbi.assigned=='Placebo')].reset_index().bbi_ci_high[0], digits)
        bbi_T = round(df_bbi.loc[(df_bbi.assigned=='Treatment')].reset_index().est[0], digits)
        bbi_T_ci_low = round(df_bbi.loc[(df_bbi.assigned=='Treatment')].reset_index().bbi_ci_low[0], digits)
        bbi_T_ci_high =round(df_bbi.loc[(df_bbi.assigned=='Treatment')].reset_index().bbi_ci_high[0], digits)

        return bbi_C, bbi_C_ci_low, bbi_C_ci_high, bbi_T, bbi_T_ci_low, bbi_T_ci_high 

    @staticmethod
    def calc_mix_cis(df_trial, n_bootstrap=100000, digits=3):

        pdf_C, pdf_T = Stats.get_mixture_distributions(df_trial)

        # the pdfs returned above have AUC=1, but for np.random we need to have sum=1
        sample_C = np.random.choice(config.doseguess_x, p=pdf_C/pdf_C.sum(), size=n_bootstrap)
        sample_T = np.random.choice(config.doseguess_x, p=pdf_T/pdf_T.sum(), size=n_bootstrap)
        bootstrap_diffs = sample_T - sample_C

        mix = round(bootstrap_diffs.mean(), digits)
        mix_ci_low = round(np.percentile(bootstrap_diffs,  2.5), digits)
        mix_ci_high = round(np.percentile(bootstrap_diffs, 97.5), digits)

        return mix, mix_ci_low, mix_ci_high

    @staticmethod
    def get_mixture_distributions(df_trial):

        pdfs={}
        for trt in ['C', 'T']:
            mixture_comps = []
            
            for row in df_trial.loc[df_trial.trt==trt].itertuples():
                mixture_comps.append(
                    scipy.stats.norm(
                        loc=row.guess_dose, 
                        scale=row.guess_sd,).pdf(config.doseguess_x))
            
            pdf = np.stack(mixture_comps, axis=0).sum(axis=0) 
            pdf = pdf / np.trapezoid(pdf, config.doseguess_x) # Normalize so AUC=1
            pdfs[trt] = pdf

        return pdfs['C'], pdfs['T']

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
        df.columns = ['assigned', 'est', 'se', 'bbi_ci_low', 'bbi_ci_high']

        return df


class Power():

    @staticmethod
    def get_df_stats(df_trials, sample_sizes, rope_cgr, rope_bbi, rope_mix):

        dfs=[]

        for scenario in df_trials.scenario.unique():

            df_stats = Power.add_cis(
                df_stats =  pd.DataFrame(), 
                df_trials = df_trials.loc[(df_trials.scenario==scenario)], 
                sample_sizes =  sample_sizes,)

            df_stats = Power.add_decisions(
                df_stats = df_stats,
                rope_cgr = rope_cgr,
                rope_bbi = rope_bbi, 
                rope_mix = rope_mix,)
            
            df_stats.insert(0, 'scenario', scenario) 
            dfs.append(df_stats)
        
        df_stats = pd.concat(dfs, ignore_index=True)
        return df_stats

    @staticmethod
    def get_df_power(df_stats):

        ### Convert equivalence test reults to numeric 
        df_stats['cgr_eqv'] = df_stats['cgr_eqv'].astype(int)
        df_stats['bbi_C_eqv'] = df_stats['bbi_C_eqv'].astype(int)
        df_stats['bbi_T_eqv'] = df_stats['bbi_T_eqv'].astype(int)
        df_stats['mix_eqv'] = df_stats['mix_eqv'].astype(int)

        ### Calculate average metric across trials
        rows = []        
        sample_sizes = df_stats.sample_size.unique()
        scenarios = df_stats.scenario.unique()

        for scenario, sample_size in product(scenarios, sample_sizes):            
            df_sample = df_stats.loc[(df_stats.scenario==scenario) & (df_stats.sample_size==sample_size)]    
            if df_sample.shape[0]==0:
                continue

            rows.append([
                scenario, # 'sample_size', 
                sample_size, # 'sample_size', 
                round(df_sample.cgr.mean(), 3), #'cgr',	
                round(df_sample.cgr_ci_low.mean(), 3), #'cgr_ci_low',
                round(df_sample.cgr_ci_high.mean(), 3), #'cgr_ci_high',
                round(df_sample.cgr_ci_high.mean()-df_sample.cgr_ci_low.mean(), 3), #'cgr_ci_mag',
                round(df_sample.cgr_eqv.mean(), 3), #'cgr_eqv', 

                round(df_sample.bbi_C.mean(), 3), #'bbi_C',
                round(df_sample.bbi_C_ci_low.mean(), 3), #'bbi_C_ci_low',
                round(df_sample.bbi_C_ci_high.mean(), 3), #'bbi_C_ci_high',
                round(df_sample.bbi_C_ci_high.mean()-df_sample.bbi_C_ci_low.mean(), 3), #'bbi_C_mag',                
                round(df_sample.bbi_C_eqv.mean(), 3), #'bbi_C_eqv', 

                round(df_sample.bbi_T.mean(), 3), #'bbi_T',
                round(df_sample.bbi_T_ci_low.mean(), 3), #'bbi_T_ci_low',
                round(df_sample.bbi_T_ci_high.mean(), 3), #'bbi_T_ci_high',
                round(df_sample.bbi_T_ci_high.mean()-df_sample.bbi_T_ci_low.mean(), 3), #'bbi_T_mag',
                round(df_sample.bbi_T_eqv.mean(), 3), #'bbi_T_eqv',                

                round(df_sample.mix.mean(), 3), #'mix',
                round(df_sample.mix_ci_low.mean(), 3), #'mix_ci_low',
                round(df_sample.mix_ci_high.mean(), 3), #'mix_ci_high',
                round(df_sample.mix_ci_high.mean()-df_sample.mix_ci_low.mean(), 3), #'mix_mag',
                round(df_sample.mix_eqv.mean(), 3), #'mix_eqv',
                ])

        ### Turn rows into df        
        df_power = pd.DataFrame(
            columns=[
                'scenario', 
                'sample_size', 
                ### Correct Guess Rate
                'cgr',	
                'cgr_ci_low',
                'cgr_ci_high',
                'cgr_ci_mag',
                'cgr_eqv', 
                ### BBI of Control
                'bbi_C',
                'bbi_C_ci_low',
                'bbi_C_ci_high',
                'bbi_C_ci_mag',
                'bbi_C_eqv', 
                ### BBI of Treatment
                'bbi_T',
                'bbi_T_ci_low',
                'bbi_T_ci_high',
                'bbi_T_ci_mag',
                'bbi_T_eqv',
                ### Mixture model
                'mix',
                'mix_ci_low',
                'mix_ci_high',
                'mix_ci_mag',
                'mix_eqv',
                ], data=rows)
       
        return df_power

    @staticmethod
    def add_cis(df_stats, df_trials, sample_sizes):

        ### Check power for different sample sizes and methods
        rows = []

        for trial, sample_size in tqdm(product(df_trials.trial.unique(), sample_sizes), desc='Calc CIs'):

            df_C = df_trials.loc[(df_trials.trial == trial) & (df_trials.trt == 'C')].reset_index().iloc[0:round(sample_size/2), :]
            df_T = df_trials.loc[(df_trials.trial == trial) & (df_trials.trt == 'T')].reset_index().iloc[0:round(sample_size/2), :]
            df_trial = pd.concat([df_C, df_T], ignore_index=True).reset_index(drop=True)

            ### Get CGR stats
            cgr, cgr_ci_low, cgr_ci_high = Stats.calc_cgr_cis(df_trial)
            mix, mix_ci_low, mix_ci_high = Stats.calc_mix_cis(df_trial)

            bbi_C, bbi_C_ci_low, bbi_C_ci_high, bbi_T, bbi_T_ci_low, bbi_T_ci_high = Stats.calc_bbi_cis(df_trial)


            rows.append({
                'trial': trial,
                'sample_size': sample_size,
                # CGR stats
                'cgr': cgr, 
                'cgr_ci_low': cgr_ci_low, 
                'cgr_ci_high': cgr_ci_high,
                # BBI stats
                'bbi_C': bbi_C, 
                'bbi_C_ci_low': bbi_C_ci_low, 
                'bbi_C_ci_high': bbi_C_ci_high,
                'bbi_T': bbi_T, 
                'bbi_T_ci_low': bbi_T_ci_low, 
                'bbi_T_ci_high': bbi_T_ci_high,
                # Mixture distribtuion stats
                'mix': mix, 
                'mix_ci_low': mix_ci_low, 
                'mix_ci_high': mix_ci_high,})

        df_stats = pd.DataFrame(rows)
        return df_stats

    @staticmethod
    def add_decisions(df_stats, rope_cgr, rope_bbi, rope_mix,):

        if all([col in df_stats.columns for col in ['cgr', 'cgr_ci_low', 'cgr_ci_high']]):            
            pos = df_stats.columns.get_loc('cgr_ci_high') + 1
            df_stats.insert(pos, 'cgr_eqv', None) 
            df_stats['cgr_eqv'] = (
                (df_stats['cgr_ci_high'] < 0.5 + rope_cgr) &
                (df_stats['cgr_ci_low']  > 0.5 - rope_cgr))

        if all([col in df_stats.columns for col in ['mix', 'mix_ci_low', 'mix_ci_high']]):
            pos = df_stats.columns.get_loc('mix_ci_high') + 1
            df_stats.insert(pos, 'mix_eqv', None) 
            df_stats['mix_eqv'] = (
                (df_stats['mix_ci_high'] < rope_mix) &
                (df_stats['mix_ci_low']  > -rope_mix))                

        if all([col in df_stats.columns for col in ['bbi_C', 'bbi_C_ci_low', 'bbi_C_ci_high']]):
            pos = df_stats.columns.get_loc('bbi_C_ci_high') + 1
            df_stats.insert(pos, 'bbi_C_eqv', None) 
            df_stats['bbi_C_eqv'] = (
                (df_stats['bbi_C_ci_high'] < rope_bbi) &
                (df_stats['bbi_C_ci_low']  > -rope_bbi))

        if all([col in df_stats.columns for col in ['bbi_T', 'bbi_T_ci_low', 'bbi_T_ci_high']]):
            pos = df_stats.columns.get_loc('bbi_T_ci_high') + 1
            df_stats.insert(pos, 'bbi_T_eqv', None) 
            df_stats['bbi_T_eqv'] = (
                (df_stats['bbi_T_ci_high'] < rope_bbi) &
                (df_stats['bbi_T_ci_low']  > -rope_bbi))

        return df_stats