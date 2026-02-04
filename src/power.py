from statsmodels.stats.proportion import proportion_confint
from scipy.stats import beta as beta_dist
from io import StringIO
from itertools import product, chain
from tqdm import tqdm
import src.config as config
import src.folders as folders
import pandas as pd
import numpy as np
import scipy
import math
import os

if 'bbi' in config.methods:
    from rpy2.robjects import r
    r('library(BI)')


class DataGeneration():

    @staticmethod
    def get_df_patientsData(scenario, n_trials, sample, params=[], digits=config.digits):

        ### Input check
        assert isinstance(scenario, str), f'scenario must be str, got {type(scenario).__name__}'
        assert isinstance(n_trials, int), f'n_trials must be int, got {type(n_trials).__name__}'
        assert isinstance(sample, int), f'sample must be int, got {type(sample).__name__}'
        assert isinstance(params, list), f'params must be dict, got {type(params).__name__}'
        assert isinstance(digits, int), f'digits must be int, got {type(digits).__name__}'
        assert sample % 2 == 0, f'sample must be even, got {sample}'

        ### Generate trial data
        df_patientsData = pd.DataFrame()
        df_patientsData['scenario'] = [scenario]*n_trials*sample
        
        trials = [[trial]*sample for trial in range(0, n_trials, 1)]
        trials = list(chain.from_iterable(trials))
        df_patientsData['trial'] = trials
        trt_per_trial = ['T']*(sample//2) + ['C']*(sample//2)
        trt = [np.random.permutation(trt_per_trial).tolist() for _ in range(n_trials)]
        df_patientsData['trt'] = list(chain.from_iterable(trt))

        df_patientsData['pID'] = [pID for pID in range(0, sample, 1)]*n_trials

        ### Add data for each param
        for param in params:

            assert isinstance(param, dict), f'each param must be dict, got {type(param).__name__}'
            
            if param['type'] == 'normal':
                if 'col' in param.keys():
                    col = param['col']
                else:
                    col = 'value'

                df_patientsData = DataGeneration.add_normal_patientData(
                    df_patientsData, 
                    param = param['arm_params'], 
                    col=col)
            
            elif param['type'] == 'binaryguess':
                if 'col' in param.keys():
                    col = param['col']
                else:
                    col = 'guess_bin'

                df_patientsData = DataGeneration.add_binary_patientData(
                    df_patientsData, 
                    param = param['arm_params'], 
                    col = col)

            else: 
                assert False

        return df_patientsData

    @staticmethod
    def add_normal_patientData(df_patientsData, param, col='value', digits=config.digits):

        # param = {
        #     'type': 'normal',
        #     'arm_params':{
        #         'C': {'mean': 10, 'sd': 11.2,},
        #         'T': {'mean': 19, 'sd': 11.2,},
        #     },
        # }

        ### Generate data for each arm 
        for arm, arm_param in param.items():     
            df_patientsData.loc[df_patientsData.trt==arm, col] = np.random.normal(
                arm_param['mean'], 
                arm_param['sd'],
                df_patientsData.loc[df_patientsData.trt==arm].shape[0])

        df_patientsData[col] = df_patientsData[col].round(digits)   
        return df_patientsData

    @staticmethod
    def add_binary_patientData(df_patientsData, param, col='guess_bin'): 

        # param = {
        #     'type': 'binaryguess',
        #     'C': {'cgr': 0.65,}
        #     'T': {'cgr': 0.65,}
        # }
 
        df_patientsData.loc[df_patientsData.trt=='C', col] = np.random.choice(
            ['C', 'T'], 
            size = df_patientsData.loc[df_patientsData.trt=='C'].shape[0],
            p = [param['C']['cgr'], 1-param['C']['cgr'],]) 

        df_patientsData.loc[df_patientsData.trt=='T', col] = np.random.choice(
            ['T', 'C'], 
            size = df_patientsData.loc[df_patientsData.trt=='T'].shape[0],
            p = [param['T']['cgr'], 1-param['T']['cgr'],]) 

        return df_patientsData


class Stats():

    @staticmethod
    def get_df_trialsResults(df_CIs, trim_CIs=True):

        df_trialsResults = df_CIs.copy()

        ### Add trial results 
        df_trialsResults = Stats.add_eqv(
            df_CIs = df_trialsResults,)

        df_trialsResults = Stats.add_nsd(
            df_CIs = df_trialsResults,)

        df_trialsResults = Stats.add_sigdiff(
            df_CIs = df_trialsResults,)

        if trim_CIs:
            rm_cols = [col for col in df_trialsResults.columns if (('_ciL' in col) | ('_ciH' in col))]
            df_trialsResults = df_trialsResults.drop(columns=rm_cols)
            
        return df_trialsResults

    @staticmethod
    def get_df_weighted_cgr(df_patientsData, col_guess='guess', col_trt='trt', col_conf='conf', digits=3):    
        """
        Calculate weighted correct guess rate for each (scenario, trial, trt) combination.
        
        Args:
            df_patientsData: DataFrame with columns 'scenario', 'trial', 'trt', 'guess', and the conf column
            col_guess: Column name containing the guessed treatment
            col_trt: Column name containing the actual treatment
            col_conf: Column name containing the weights (e.g., confidence scores)
            digits: Number of decimal places to round results
            
        Returns:
            DataFrame with columns: scenario, trial, trt, cgr (weighted correct guess rate), se, ciL, ciH
        """
        scenarios = df_patientsData.scenario.unique().tolist()
        trials = df_patientsData.trial.unique().tolist()
        trts = df_patientsData.trt.unique().tolist() + ['all']

        rows=[]
        for scenario, trial, trt in product(scenarios, trials, trts):

            if trt=='all':
                df_tmp = df_patientsData.loc[
                    (df_patientsData.scenario==scenario) & 
                    (df_patientsData.trial==trial)]
            else:
                df_tmp = df_patientsData.loc[
                    (df_patientsData.scenario==scenario) & 
                    (df_patientsData.trial==trial) & 
                    (df_patientsData.trt==trt)]

            if len(df_tmp) == 0:
                continue

            match = (df_tmp[col_guess] == df_tmp[col_trt]).astype(float).to_numpy()
            w = pd.to_numeric(df_tmp[col_conf], errors='coerce').to_numpy(dtype=float)
            wcgr = np.average(match, weights=w) # weighted correct guess rate        
            n_eff = (np.sum(w) ** 2) / np.sum(w ** 2) # Kish's effective sample size
            ciL, ciH = proportion_confint(wcgr*n_eff, n_eff)    

            row={}
            row['scenario'] = scenario
            row['trial'] = trial
            row['trt'] = trt
            row['wcgr'] = round(wcgr, digits)
            row['ciL'] = round(ciL, digits)
            row['ciH'] = round(ciH, digits)
            row['MoE'] = round((ciH-ciL)/2, digits)
            row['n'] = len(df_tmp)
            row['n_eff'] = round(n_eff, digits)
            rows.append(row)
        
        df = pd.DataFrame(rows) 
        return df

    @staticmethod
    def get_df_weighted_gmgs(df_patientsData, col_value='value', col_conf='conf', digits=2):    
        """
        Calculate weighted mean and SD of values for each (scenario, trial, trt) combination.
        
        Args:
            df_patientsData: DataFrame with columns 'scenario', 'trial', 'trt', and the value/conf columns
            col_value: Column name containing the values to average
            col_conf: Column name containing the weights (e.g., confidence scores)
            digits: Number of decimal places to round results
            
        Returns:
            DataFrame with columns: scenario, trial, trt, w_mean, w_sd
        """
        scenarios = df_patientsData.scenario.unique()
        trials = df_patientsData.trial.unique()
        trts = df_patientsData.trt.unique()

        rows=[]
        for scenario, trial, trt in product(scenarios, trials, trts):

            df_tmp = df_patientsData.loc[
                (df_patientsData.scenario==scenario) & 
                (df_patientsData.trial==trial) & 
                (df_patientsData.trt==trt)]

            row={}
            row['scenario'] = scenario
            row['trial'] = trial
            row['trt'] = trt

            x = pd.to_numeric(df_tmp[col_value], errors='coerce').to_numpy(dtype=float)
            w = pd.to_numeric(df_tmp[col_conf], errors='coerce').to_numpy(dtype=float)

            mu = np.average(x, weights=w)
            var = np.average((x - mu) ** 2, weights=w)
            sd = math.sqrt(var)            
            
            # Using here Kish's effective sample size We have reliability weights, reflecting precision/confidence, 
            # not count (and hence not just sum of weights)
            n_eff = (np.sum(w) ** 2) / np.sum(w ** 2) 
            
            se = sd / math.sqrt(n_eff)
            t_crit = scipy.stats.t.ppf(0.975, df=max(n_eff - 1, 1))

            row[col_value] = round(mu, digits)
            row['sd'] = round(sd, digits)
            row['se'] = round(se, digits)
            row['ciL'] = round(mu - se * t_crit, digits)
            row['ciH'] = round(mu + se * t_crit, digits)
            row['moe'] = round(se * t_crit, digits)
            rows.append(row)
        
        df = pd.DataFrame(rows) 
        df = df[['scenario', 'trial', 'trt', 'gmg', 'ciL', 'ciH', 'moe']]
        return df    

    @staticmethod
    def get_df_combinedCI(df_CIs: pd.DataFrame, col_value='value', col_ciL='ciL', col_ciH='ciH', col_n='n', alpha=0.05, digits=2):
        """
        Combine multiple CIs into a single CI as if the underlying data were pooled.
        Assumes all CIs are t-based with known sample sizes.
        """
        
        scenarios = df_CIs.scenario.unique()
        trials = df_CIs.trial.unique()
        trts = df_CIs.trt.unique()

        rows=[]
        for scenario, trial, trt in product(scenarios, trials, trts):

            df_tmp = df_CIs.loc[
                (df_CIs.scenario==scenario) & 
                (df_CIs.trial==trial) & 
                (df_CIs.trt==trt)]

            ciL = df_tmp[col_ciL].to_numpy(dtype=float)
            ciH = df_tmp[col_ciH].to_numpy(dtype=float)

            if col_n in df_tmp.columns:
                n = df_tmp[col_n].to_numpy(dtype=float)
                t_crits = scipy.stats.t.ppf(1 - alpha/2, df=n - 1) 
            else:
                # Note: if sample sizes are unknown, then we are pretending n=2 for all rows. 
                # This is because in the 'guess conf-to-CI' approach, CIs are constructed for each individual, but the t-distribution is not defined for n=1.
                # This is a price we pay converting guess confidence from single individuals to confidence interval.                 
                n = np.ones(df_tmp.shape[0])
                t_crits = scipy.stats.t.ppf(1 - alpha/2, df=1)

            # Recover mean and SD from each CI
            n_total = n.sum()
            means = (ciL + ciH) / 2
            ses = (ciH - ciL) / (2 * t_crits)
            sds = ses * np.sqrt(n)
            
            # Pooled mean
            mean_pooled = (n * means).sum() / n_total
            
            # Pooled variance: combines within-group and between-group variance
            ss_within = ((n - 1) * sds**2).sum()
            ss_between = (n * (means - mean_pooled)**2).sum()
            var_pooled = (ss_within + ss_between) / (n_total - 1)
            
            # Combined CI
            se_pooled = np.sqrt(var_pooled / n_total)
            t_crit = scipy.stats.t.ppf(1 - alpha/2, df=n_total - 1)

            row={}
            row['scenario'] = scenario
            row['trial'] = trial
            row['trt'] = trt
            row['n'] = n_total
            row[col_value] = float(round(mean_pooled, digits))
            row['ciL'] = float(round(mean_pooled - t_crit * se_pooled, digits))
            row['ciH'] = float(round(mean_pooled + t_crit * se_pooled, digits))
            row['moe'] = float(round((row['ciH']-row['ciL'])/2, digits))

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def get_df_combinedCI_fixed(df_CIs: pd.DataFrame, col_value='value', col_ciL='ciL', col_ciH='ciH', col_n='n', alpha=0.05, digits=2):
        """
        Combine multiple CIs into a single CI as if the underlying data were pooled.
        Assumes all CIs are t-based with known sample sizes.
        """
        
        scenarios = df_CIs.scenario.unique()
        trials = df_CIs.trial.unique()
        trts = df_CIs.trt.unique()

        rows=[]
        for scenario, trial, trt in product(scenarios, trials, trts):

            df_tmp = df_CIs.loc[
                (df_CIs.scenario==scenario) & 
                (df_CIs.trial==trial) & 
                (df_CIs.trt==trt)]

            ciL = df_tmp[col_ciL].to_numpy(dtype=float)
            ciH = df_tmp[col_ciH].to_numpy(dtype=float)

            # Note: we are pretending n=2, while its n=1, but otherwise the t-distribution is not defined.
            # This is a price we pay converting guess confidence from single individuals to confidence interval. 
            n = df_tmp[col_n].to_numpy(dtype=float) if col_n in df_tmp.columns else 2*np.ones(len(df_tmp))
        
            # Recover mean and SD from each CI
            t_crits = scipy.stats.t.ppf(1 - alpha/2, df=n - 1)
            means = (ciL + ciH) / 2
            ses = (ciH - ciL) / (2 * t_crits)
            sds = ses * np.sqrt(n)
            
            # Pooled mean
            n_total = n.sum()
            mean_pooled = (n * means).sum() / n_total
            
            # Pooled variance: combines within-group and between-group variance
            ss_within = ((n - 1) * sds**2).sum()
            ss_between = (n * (means - mean_pooled)**2).sum()
            var_pooled = (ss_within + ss_between) / (n_total - 1)
            
            # Combined CI
            se_pooled = np.sqrt(var_pooled / n_total)
            t_crit = scipy.stats.t.ppf(1 - alpha/2, df=n_total - 1)

            row={}
            row['scenario'] = scenario
            row['trial'] = trial
            row['trt'] = trt
            row[col_value] = float(round(mean_pooled, digits))
            row['ciL'] = float(round(mean_pooled - t_crit * se_pooled, digits))
            row['ciH'] = float(round(mean_pooled + t_crit * se_pooled, digits))
            row['moe'] = float(round((row['ciH']-row['ciL'])/2, digits))

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def get_df_differenceCI(df_CIs: pd.DataFrame, col_value='value', col_ciL='ciL', col_ciH='ciH', col_n='n', alpha=0.05, digits=2):
        """
        Combine multiple CIs into a single CI as if the underlying data were pooled.
        Assumes all CIs are t-based with known sample sizes.
        """
        
        scenarios = df_CIs.scenario.unique()
        trials = df_CIs.trial.unique()
        trts = sorted(df_CIs.trt.unique())
        assert len(trts) == 2

        rows=[]
        for scenario, trial in product(scenarios, trials):

            df_tmp = df_CIs.loc[
                (df_CIs.scenario==scenario) & 
                (df_CIs.trial==trial)]

            # Split by treatment group
            df_C = df_tmp[df_tmp['trt'] == trts[0]]
            df_T = df_tmp[df_tmp['trt'] == trts[1]]

            mC, vC, nC = Helpers.recover_stats(df_C, col_ciL, col_ciH, col_n, alpha)
            mT, vT, nT = Helpers.recover_stats(df_T, col_ciL, col_ciH, col_n, alpha)

            # Now match ttest_ind: pooled variance across both groups, CI of difference
            df_dof = nC + nT - 2
            sp2 = ((nC - 1) * vC + (nT - 1) * vT) / df_dof  # pooled variance
            se_diff = np.sqrt(sp2 * (1/nC + 1/nT))

            diff = mT - mC
            t_crit = scipy.stats.t.ppf(1 - alpha/2, df=df_dof)

            row = {}
            row['scenario'] = scenario
            row['trial'] = trial
            row[col_value] = float(round(diff, digits))
            row['ciL'] = float(round(diff - t_crit * se_diff, digits))
            row['ciH'] = float(round(diff + t_crit * se_diff, digits))
            row['moe'] = float(round(t_crit * se_diff, digits))
            rows.append(row)

        return pd.DataFrame(rows)


    ''' Calc CIs '''
    @staticmethod
    def get_df_diffCIs(df_patientsData, samples=[100], df_CIs=pd.DataFrame(), digits=config.digits):

        scenarios = df_patientsData.scenario.unique()
        trials = df_patientsData.trial.unique()
        rows=[]

        for scenario, trial, sample in tqdm(product(scenarios, trials, samples), desc='Calc CIs'):

            df = df_patientsData.loc[
                (df_patientsData.scenario==scenario) & 
                (df_patientsData.trial==trial)].iloc[0:sample, :]

            assert df.shape[0] == sample

            values_C = df.loc[df.trt=='C'].value
            values_T = df.loc[df.trt=='T'].value

            ttest_res = scipy.stats.ttest_ind(values_T, values_C)
            ciL = ttest_res.confidence_interval().low
            ciH = ttest_res.confidence_interval().high

            row={}
            row['scenario'] = scenario
            row['trial'] = trial
            row['sample'] = sample
            row['diff'] = values_T.mean() - values_C.mean()
            row['ciL'] = ciL
            row['ciH'] = ciH
            rows.append(row)

        # Housekeeping
        df_diffCI = pd.DataFrame(rows)
        df_diffCI['moe'] = (df_diffCI['ciH'] - df_diffCI['ciL'])/2
        df_diffCI['diff'] = df_diffCI['diff'].round(digits)
        df_diffCI['ciL'] = df_diffCI['ciL'].round(digits)
        df_diffCI['ciH'] = df_diffCI['ciH'].round(digits)
        df_diffCI['moe'] = df_diffCI['moe'].round(digits)

        df_CIs = pd.concat([df_CIs, df_diffCI], ignore_index=True)
        return df_CIs

    @staticmethod
    def get_df_diffCIs_vector(df_patientsData, col='value', samples=[100], df_CIs=pd.DataFrame(), digits=config.digits, alpha=0.05):
        df = df_patientsData.copy()
        df['row_num'] = df.groupby(['scenario', 'trial']).cumcount()

        rows = []
        for sample in samples:
            df_sample = df[df['row_num'] < sample]

            grouped = (
                df_sample
                .groupby(['scenario', 'trial', 'trt'], as_index=False)[col]
                .agg(n='count', mean='mean', var=lambda s: s.var(ddof=1))
            )

            total_n = grouped.groupby(['scenario', 'trial'], as_index=False)['n'].sum().rename(columns={'n': 'total_n'})
            grouped = grouped.merge(total_n, on=['scenario', 'trial'], how='left')
            grouped = grouped[grouped['total_n'] == sample]
            grouped = grouped.drop(columns=['total_n'])
            grouped['sample'] = sample
            rows.append(grouped)

        if len(rows) == 0:
            return df_CIs

        g = pd.concat(rows, ignore_index=True)

        p = g.pivot(index=['scenario', 'trial', 'sample'], columns='trt', values=['n', 'mean', 'var'])

        nC = p.get(('n', 'C'), pd.Series(index=p.index, dtype=float)).to_numpy(dtype=float)
        nT = p.get(('n', 'T'), pd.Series(index=p.index, dtype=float)).to_numpy(dtype=float)
        mC = p.get(('mean', 'C'), pd.Series(index=p.index, dtype=float)).to_numpy(dtype=float)
        mT = p.get(('mean', 'T'), pd.Series(index=p.index, dtype=float)).to_numpy(dtype=float)
        vC = p.get(('var', 'C'), pd.Series(index=p.index, dtype=float)).to_numpy(dtype=float)
        vT = p.get(('var', 'T'), pd.Series(index=p.index, dtype=float)).to_numpy(dtype=float)

        # Match scipy.stats.ttest_ind default: equal_var=True (pooled variance), two-sided CI
        df_dof = (nC + nT - 2.0)
        valid = np.isfinite(nC) & np.isfinite(nT) & np.isfinite(vC) & np.isfinite(vT) & (nC >= 2) & (nT >= 2) & (df_dof > 0)

        ciL = np.full(df_dof.shape, np.nan, dtype=float)
        ciH = np.full(df_dof.shape, np.nan, dtype=float)

        if np.any(valid):
            sp2 = ((nC[valid] - 1.0) * vC[valid] + (nT[valid] - 1.0) * vT[valid]) / df_dof[valid]
            se = np.sqrt(sp2 * (1.0 / nC[valid] + 1.0 / nT[valid]))

            diff = mT[valid] - mC[valid]
            tcrit = scipy.stats.t.ppf(1.0 - alpha / 2.0, df_dof[valid])

            ciL[valid] = diff - tcrit * se
            ciH[valid] = diff + tcrit * se

        df_diffCIs = p.reset_index()[['scenario', 'trial', 'sample']].assign(diff=diff, ciL=ciL, ciH=ciH)

        # Housekeeping
        df_diffCIs.columns = df_diffCIs.columns.droplevel(1)

        df_diffCIs['moe'] = (df_diffCIs['ciH'] - df_diffCIs['ciL']) / 2
        df_diffCIs['diff'] = df_diffCIs['diff'].round(digits)
        df_diffCIs['ciL'] = df_diffCIs['ciL'].round(digits)
        df_diffCIs['ciH'] = df_diffCIs['ciH'].round(digits)
        df_diffCIs['moe'] = df_diffCIs['moe'].round(digits)
        df_diffCIs = df_diffCIs.rename(columns={
            'diff': f'{col}_diff',
            'ciL': f'{col}_ciL',
            'ciH': f'{col}_ciH',
            'moe': f'{col}_moe',})    

        df_CIs = pd.concat([df_CIs, df_diffCIs], ignore_index=True)
        return df_CIs

    @staticmethod
    def get_df_cgrCIs(df_patientsData, samples=[100], df_CIs=pd.DataFrame(), digits=config.digits):

        scenarios = df_patientsData.scenario.unique()
        trials = df_patientsData.trial.unique()
        rows=[]

        for scenario, trial, sample in tqdm(product(scenarios, trials, samples), desc='Calc CIs'):

            df = df_patientsData.loc[
                (df_patientsData.scenario==scenario) & 
                (df_patientsData.trial==trial)].iloc[0:sample, :]
            
            matches = (df['trt'] == df['guess_bin']).astype(int)
            k = matches.sum()
            n = len(matches)
            cgr = k / n
            cgr_ciL, cgr_ciH = proportion_confint(k, n, alpha=0.05, method="beta")

            row={}
            row['scenario'] = scenario
            row['trial'] = trial
            row['sample'] = sample
            row['cgr'] = cgr
            row['cgr_ciL'] = cgr_ciL
            row['cgr_ciH'] = cgr_ciH
            rows.append(row)

        ### Housekeeping
        df_cgr_CIs = pd.DataFrame(rows)
        df_cgr_CIs['cgr_moe'] = (df_cgr_CIs['cgr_ciH'] - df_cgr_CIs['cgr_ciL'])/2
        df_cgr_CIs['cgr'] = df_cgr_CIs['cgr'].round(digits)
        df_cgr_CIs['cgr_ciH'] = df_cgr_CIs['cgr_ciH'].round(digits)
        df_cgr_CIs['cgr_ciL'] = df_cgr_CIs['cgr_ciL'].round(digits)
        df_cgr_CIs['cgr_moe'] = df_cgr_CIs['cgr_moe'].round(digits)
        
        df_CIs = pd.concat([df_CIs, df_cgr_CIs], ignore_index=True)
        df_CIs = df_CIs.sort_values(by=['scenario', 'trial', 'sample',], ascending=True, ignore_index=True)

        return df_CIs

    @staticmethod
    def get_df_cgrCIs_vector(df_patientsData, samples=[100], df_CIs=pd.DataFrame(), digits=config.digits):
        """ Vectorized CGR CI calculation - much faster than loop-based approaches"""
        
        ### Add match column (trt == guess_bin)
        df = df_patientsData.copy()
        df['match'] = (df['trt'] == df['guess_bin']).astype(int)
        
        ### Add row number within each (scenario, trial) group for sampling
        df['row_num'] = df.groupby(['scenario', 'trial']).cumcount()
        
        rows = []
        for sample in samples:
            ### Filter to first sample rows per (scenario, trial)
            df_sample = df[df['row_num'] < sample]
            
            ### Group by (scenario, trial) and compute k, n
            grouped = df_sample.groupby(['scenario', 'trial']).agg(
                k=('match', 'sum'),
                n=('match', 'count')
            ).reset_index()
            
            ### Only keep groups with exactly sample rows (to match reference assert)
            grouped = grouped[grouped['n'] == sample]
            grouped['sample'] = sample
            rows.append(grouped)
        
        if len(rows) == 0:
            return df_CIs
        
        ### Combine all sample sizes
        df_cgrCIs = pd.concat(rows, ignore_index=True)
        
        ### Vectorized CI calculation using beta distribution
        k = df_cgrCIs['k'].values
        n = df_cgrCIs['n'].values
        alpha = 0.05
        
        df_cgrCIs['cgr'] = k / n
        df_cgrCIs['cgr_ciL'] = beta_dist.ppf(alpha / 2, k, n - k + 1)
        df_cgrCIs['cgr_ciH'] = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
        
        ### Handle edge cases (k=0 or k=n)
        df_cgrCIs.loc[k==0, 'cgr_ciL'] = 0.0
        df_cgrCIs.loc[k==n, 'cgr_ciH'] = 1.0
        
        ### Housekeeping
        df_cgrCIs['cgr_moe'] = (df_cgrCIs['cgr_ciH'] - df_cgrCIs['cgr_ciL'])/2
        df_cgrCIs['cgr'] = df_cgrCIs['cgr'].round(digits)
        df_cgrCIs['cgr_ciH'] = df_cgrCIs['cgr_ciH'].round(digits)
        df_cgrCIs['cgr_ciL'] = df_cgrCIs['cgr_ciL'].round(digits)
        df_cgrCIs['cgr_moe'] = df_cgrCIs['cgr_moe'].round(digits)
        
        df_CIs = pd.concat([df_CIs, df_cgrCIs[['scenario', 'trial', 'sample', 'cgr', 'cgr_ciL', 'cgr_ciH', 'cgr_moe']]], ignore_index=True)
        df_CIs = df_CIs.sort_values(by=['scenario', 'trial', 'sample',], ascending=True, ignore_index=True)

        return df_CIs


    ''' Decisions '''
    @staticmethod
    def add_sigdiff(df_CIs, thresholds={'cgr': 0.5}):

        for col in Helpers.find_ci_column_labels(df_CIs):

            if col in thresholds.keys():
                thr = thresholds[col]
            else:
                thr=0

            if f'{col}_moe' in df_CIs.columns:
                pos = df_CIs.columns.get_loc(f'{col}_moe') + 1
            else:
                pos = df_CIs.columns.get_loc(f'{col}_ciH') + 1

            df_CIs.insert(pos, f'{col}_sigdiff', None) 
            df_CIs[f'{col}_sigdiff'] = (
                (df_CIs[f'{col}_ciL'] > thr) | 
                (df_CIs[f'{col}_ciH'] < thr))

            df_CIs[f'{col}_sigdiff'] = df_CIs[f'{col}_sigdiff'].astype(bool)

        return df_CIs      
        
    @staticmethod
    def add_nsd(df_CIs, thresholds={'cgr': 0.5}):


        for col in Helpers.find_ci_column_labels(df_CIs):

            if col in thresholds.keys():
                thr = thresholds[col]
            else:
                thr = 0

            if f'{col}_moe' in df_CIs.columns:
                pos = df_CIs.columns.get_loc(f'{col}_moe') + 1
            else:
                pos = df_CIs.columns.get_loc(f'{col}_ciH') + 1

            df_CIs.insert(pos, f'{col}_nsd', None) 
            df_CIs[f'{col}_nsd'] = (
                (df_CIs[f'{col}_ciL'] < thr) &
                (df_CIs[f'{col}_ciH'] > thr))

            df_CIs[f'{col}_nsd'] = df_CIs[f'{col}_nsd'].astype(bool)

        return df_CIs    


class Power():

    @staticmethod
    def get_df_power(df_trialsResults, digits=config.digits, methods=config.methods):

        df_trialsResults = Helpers.convert_res_to_numeric(df_trialsResults)
        samples = df_trialsResults.sample.unique()
        scenarios = df_trialsResults.scenario.unique()
        rows = []        

        ### Calculate average across trials for each scenario / sample size
        for scenario, sample in tqdm(product(scenarios, samples), desc='Calc df_power'):            

            df_sample = df_trialsResults.loc[(df_trialsResults.scenario==scenario) & (df_trialsResults.sample==sample)]    
            if df_sample.shape[0]==0:
                continue

            row={}
            row['scenario'] = scenario
            row['sample'] = sample
    
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
        cols_to_round.remove('sample')
        df_power[cols_to_round] = df_power[cols_to_round].round(digits)        

        return df_power


class Helpers():

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

    @staticmethod
    def find_ci_column_labels(df: pd.DataFrame) -> list[str]:
        """
        Find column labels that have both _ciL and _ciH columns in the DataFrame.
        
        Args:
            df: pandas DataFrame to search
            
        Returns:
            List of label strings where both f'{label}_ciL' and f'{label}_ciH' exist
        """
        columns = set(df.columns)
        labels = set()
        
        for col in columns:
            if col.endswith('_ciL'):
                label = col[:-4]  # Remove '_ciL' suffix
                if f'{label}_ciH' in columns:
                    labels.add(label)
        
        return list(labels)        

    @staticmethod
    def get_pop_data():
        ''' Get POP data in df_patientsData format. 
            Path to data is intentionally hard coded as it is not meant to be shared. 
        
        Returns:
            df_pop: pandas DataFrame with POP data in df_patientsData format
        '''

        df_pop = pd.read_csv('C:\\Users\\szb37\\My Drive\\Projects\\POP\\codebase\\exports POP\\pop_master.csv')
        df_pop = df_pop.loc[
            (df_pop.measure.isin(['TRTGUESS_pt_dose', 'TRTGUESS_pt_conf'])) &
            (df_pop.tp=='A0')]

        df_pop = df_pop[['pID', 'condition', 'measure', 'score']].reset_index(drop=True)
        df_pop = df_pop.replace({
            'condition': {'Control': 'C', 'Treatment': 'T'},
            'measure': {'TRTGUESS_pt_dose': 'gmg', 'TRTGUESS_pt_conf': 'conf'}})
        df_pop = df_pop.rename(columns={'condition': 'trt'})

        df_pop = df_pop.pivot(index=['pID', 'trt'], columns='measure', values='score').reset_index()
        df_pop.columns.name = None

        # Adding in columns to make it like df_patientsData
        df_pop['scenario'] = 'POP'
        df_pop['trial'] = 0

        df_pop['gmg_se'] = df_pop['conf'].map(config.conf_to_se)
        df_pop['gmg_sd'] = df_pop['gmg_se'] # for sample=1, SE=SD
        df_pop['gmg_ciL'] = df_pop['gmg']-1.96*df_pop['gmg_se'] 
        df_pop['gmg_ciH'] = df_pop['gmg']+1.96*df_pop['gmg_se']
        df_pop = df_pop.round({'gmg': 2, 'gmg_sd': 2, 'gmg_se': 2, 'gmg_ciL': 2, 'gmg_ciH': 2,})
        df_pop = df_pop[['scenario', 'trial', 'pID', 'trt', 'gmg', 'conf', 'gmg_se', 'gmg_ciL', 'gmg_ciH']]

        return df_pop      

    @staticmethod
    def save_fig(fig, fname):
        for format in ['png', 'svg']:
            fig.savefig(
                fname=os.path.join(folders.powerplots, f'{fname}.{format}'),
                bbox_inches='tight',
                format=format,
                dpi=300,)          

    @staticmethod
    def recover_stats(df, col_ciL='ciL', col_ciH='ciH', col_n='n', alpha=0.05):
        """Recover mean, variance, and n from CIs"""
        ciL = df[col_ciL].to_numpy(dtype=float)
        ciH = df[col_ciH].to_numpy(dtype=float)
        n = df[col_n].to_numpy(dtype=float)
        
        t_crits = scipy.stats.t.ppf(1 - alpha/2, df=n - 1)
        means = (ciL + ciH) / 2
        ses = (ciH - ciL) / (2 * t_crits)
        sds = ses * np.sqrt(n)
        
        # Pool within each group first
        n_total = n.sum()
        mean_pooled = (n * means).sum() / n_total
        
        ss_within = ((n - 1) * sds**2).sum()
        ss_between = (n * (means - mean_pooled)**2).sum()
        var_pooled = (ss_within + ss_between) / (n_total - 1)
        
        return mean_pooled, var_pooled, n_total
