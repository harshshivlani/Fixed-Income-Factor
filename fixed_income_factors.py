import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import edhec_risk_kit as erk
import edhec_risk_kit_206 as erk1
import yfinance as yf
import statsmodels.api as sm
import seaborn as sns
from datetime import date
from pandas_datareader import data
import pandas_datareader.data as web
import quantstats as qs
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import quandl
import plotly.express as px
import plotly.graph_objects as go
import plotly

quandl.ApiConfig.api_key = 'KZ69tzkHfXscfQ1qcJ5K'



### Momentum Factor Backtest

def momentum_factor_backtest_monthly(price_data, rebal, lookback_period, n_securities, long_short, sample_start, sample_end):
    """
    Backtest Momentum Factor based on monthly rebalancing and lookback period provided.
    Capaable of long/short and long only monthtly portfolio returns choosing top 1 to top 3 and bottom 1 to bottom 3 securities
    """
    mom = price_data.pct_change(lookback_period).dropna()
    
    mom.columns= ['IG 1-3Y', 'IG 3-5Y', 'IG 7-10Y', 'US HY', 'Crossover', 'EM HY',
       '1-3Y UST', 'Intermediate UST', '7-10Y UST', 'Long Term UST']
    price_data.columns = ['IG 1-3YM', 'IG 3-5YM', 'IG 7-10YM', 'US HYM', 'CrossoverM', 'EM HYM',
       '1-3Y USTM', 'Intermediate USTM', '7-10Y USTM', 'Long Term USTM']

    data = mom.merge(price_data, on='DATE')
    
    #Convert Data Frequency to Rebalancing Frequency
    #data = data.asfreq(""+str(rebalancing_period)+"D")
    if rebal =='Monthly':
        month1 = pd.Series(data.index.month)
        month2 = pd.Series(data.index.month).shift(-1)
    elif rebal =='Quarterly':
        month1 = pd.Series(data.index.quarter)
        month2 = pd.Series(data.index.quarter).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    rets = data['Crossover'].copy()*0
    rets.name = 'Momentum Strategy'
    
    if long_short == 'No':
        for i in range(len(data)-1):
            if n_securities == 1:
                rets[i+1] = data[str(data.iloc[i,:10].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].idxmax())+'M'][i]-1
            elif n_securities == 2:
                rets[i+1] = (data[str(data.iloc[i,:10].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].idxmax())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'M'][i]-1)/2
            elif n_securities == 3:
                rets[i+1] = (data[str(data.iloc[i,:10].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].idxmax())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:8].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[:8].idxmax())+'M'][i]-1)/3
    
    if long_short == 'Yes':
        for i in range(len(data)-1):
            if n_securities == 1:
                rets[i+1] = data[str(data.iloc[i,:10].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].idxmax())+'M'][i]-1 - data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'M'][i]-1
            elif n_securities == 2:
                rets[i+1] = (data[str(data.iloc[i,:10].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].idxmax())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'M'][i]-1)/2 - (data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[1:].idxmin())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[1:].idxmin())+'M'][i]-1)/2
            elif n_securities == 3:
                rets[i+1] = (data[str(data.iloc[i,:10].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].idxmax())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:8].idxmax())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[:8].idxmax())+'M'][i]-1)/3 - (data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[1:].idxmin())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[1:].idxmin())+'M'][i]-1 + data[str(data.iloc[i,:10].sort_values()[2:].idxmin())+'M'][i+1]/data[str(data.iloc[i,:10].sort_values()[2:].idxmin())+'M'][i]-1)/3
    
    
    #Merge Value Factor Returns Data with original data and other individual securities returns 
    data = data.merge(rets, on='DATE')
    data.columns = ['IG 1-3 Yield', 'IG 3-5 Yield', 'IG 7-10 Yield', 'US HY Yield',
       'Crossover Yield', 'EM High Yield', 'UST 1-3 Yield', 'UST Int Yield',
       'UST 7-10 Yield', 'UST Long Yield', 'IG 1-3', 'IG 3-5', 'IG 7-10', 'US HY',
       'Crossover', 'EM HY', 'UST 1-3', 'UST Int',
       'UST 7-10', 'UST Long', 'Momentum Strategy']
    m_rets = data[['IG 1-3', 'IG 3-5', 'IG 7-10', 'US HY', 'Crossover', 'EM HY', 'UST 1-3', 'UST Int', 'UST 7-10', 'UST Long']].pct_change().dropna().merge(rets, on='DATE')
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    m_rets['EW'] = m_rets[['IG 1-3', 'IG 3-5', 'IG 7-10', 'US HY', 'Crossover', 'EM HY', 'UST 1-3', 'UST Int', 'UST 7-10', 'UST Long']].mean(axis=1)
    
    return m_rets


#### Value Factor Backtest        
        
def value_factor_backtest_monthly(real_yields, price_data, zscore_lookback, z_score_smoothening, n_securities, long_short, sample_start, sample_end):
    """
    Returns the Monthly Returns of Long Only Value Factor from a given set of bond index returns and real yields by going 
    long on the highest Z-Score Real Yield of the all bond indices every month end
    
    """
    #Calculate Z-Score of Real Yields
    ry_zscore = (real_yields - real_yields.rolling(260*zscore_lookback).mean())/real_yields.rolling(260*zscore_lookback).std(ddof=1)
    ry_zscore = ry_zscore.dropna().rolling(z_score_smoothening).mean()
    
    #Merge Z-Score & Total Return Indices Data
    data = ry_zscore.merge(price_data, on='DATE').dropna()
    
    #Convert Data Frequency to Rebalancing Frequency
    #data = data1.asfreq(""+str(rebalancing_period)+"D")
    month1 = pd.Series(data.index.month)
    month2 = pd.Series(data.index.month).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    #Rename Columns for better understanding
    data.columns = ['IG 1-3 Yield', 'IG 3-5 Yield', 'IG 7-10 Yield', 'US HY Yield',
       'Crossover Yield', 'EM High Yield', 'UST 1-3 Yield', 'UST Int Yield',
       'UST 7-10 Yield', 'UST Long Yield', 'IG 1-3 YieldP', 'IG 3-5 YieldP', 'IG 7-10 YieldP', 'US HY YieldP',
       'Crossover YieldP', 'EM High YieldP', 'UST 1-3 YieldP', 'UST Int YieldP',
       'UST 7-10 YieldP', 'UST Long YieldP']
    
    #Calculate Backtest Returns based on Long/Short or Long/Only and the no. of securities
    rets = data['Crossover Yield'].copy()*0
    rets.name = 'Value Strategy'
    
    if long_short == 'No':
        for i in range(len(data)-1):
            if n_securities == 1:
                rets[i+1] = data[str(data.iloc[i,:10].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].idxmax())+'P'][i]-1
            elif n_securities == 2:
                rets[i+1] = (data[str(data.iloc[i,:10].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].idxmax())+'P'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'P'][i]-1)/2
            elif n_securities == 3:
                rets[i+1] = (data[str(data.iloc[i,:10].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].idxmax())+'P'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'P'][i]-1 + + data[str(data.iloc[i,:10].sort_values()[:8].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[:8].idxmax())+'P'][i]-1)/3
    
    if long_short == 'Yes':
        for i in range(len(data)-1):
            if n_securities == 1:
                rets[i+1] = data[str(data.iloc[i,:10].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].idxmax())+'P'][i]-1 - data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'P'][i]-1
            elif n_securities == 2:
                rets[i+1] = (data[str(data.iloc[i,:10].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].idxmax())+'P'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'P'][i]-1)/2 - (data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'P'][i]-1 + data[str(data.iloc[i,:10].sort_values()[1:].idxmin())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[1:].idxmin())+'P'][i]-1)/2
            elif n_securities == 3:
                rets[i+1] = (data[str(data.iloc[i,:10].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].idxmax())+'P'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[:9].idxmax())+'P'][i]-1 + data[str(data.iloc[i,:10].sort_values()[:8].idxmax())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[:8].idxmax())+'P'][i]-1)/3 - (data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[0:].idxmin())+'P'][i]-1 + data[str(data.iloc[i,:10].sort_values()[1:].idxmin())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[1:].idxmin())+'P'][i]-1 + data[str(data.iloc[i,:10].sort_values()[2:].idxmin())+'P'][i+1]/data[str(data.iloc[i,:10].sort_values()[2:].idxmin())+'P'][i]-1)/3
    
    
    #Merge Value Factor Returns Data with original data and other individual securities returns 
    data = data.merge(rets, on='DATE')
    data.columns = ['IG 1-3 Yield', 'IG 3-5 Yield', 'IG 7-10 Yield', 'US HY Yield',
       'Crossover Yield', 'EM High Yield', 'UST 1-3 Yield', 'UST Int Yield',
       'UST 7-10 Yield', 'UST Long Yield', 'IG 1-3', 'IG 3-5', 'IG 7-10', 'US HY',
       'Crossover', 'EM HY', 'UST 1-3', 'UST Int',
       'UST 7-10', 'UST Long', 'Value Strategy']
    m_rets = data[['IG 1-3', 'IG 3-5', 'IG 7-10', 'US HY', 'Crossover', 'EM HY', 'UST 1-3', 'UST Int', 'UST 7-10', 'UST Long']].pct_change().dropna().merge(rets, on='DATE')
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    m_rets['EW'] = m_rets[['IG 1-3', 'IG 3-5', 'IG 7-10', 'US HY', 'Crossover', 'EM HY', 'UST 1-3', 'UST Int', 'UST 7-10', 'UST Long']].mean(axis=1)
    
    return m_rets






def momentum_factor_backtest(price_data, rebal, lookback_period, duration_adj, n_securities, long_short, sample_start, sample_end, cost, spread=0, lev_factor=1, trans_cost=0.00):
    """
    Backtest Momentum Factor based on month end rebalancing and lookback period provided. Factor is adjusted for duration by dividing lookback percentage return with weighted average duration of fund/index.
    Capable of long/short and long only daily portfolio returns choosing top 1 to top 3 and bottom 1 to bottom 3 securities
    """
    if duration_adj== 'Yes':
        mom = price_data.pct_change(lookback_period).dropna()/[1,2,4,8,25,10,10,8,3,2,5,24,10,5,4,9,6,6,6,4,8,4]
    else:
        mom = price_data.pct_change(lookback_period).dropna()
    
    cols = list(price_data.columns)
    mom.columns= price_data.columns
    price_data.columns = price_data.columns + str('M')

    data = mom.merge(price_data, on='Date')
    
    #Convert Data Frequency to Rebalancing Frequency
    #data = data.asfreq(""+str(rebalancing_period)+"D")
    if rebal =='Monthly':
        month1 = pd.Series(data.index.month)
        month2 = pd.Series(data.index.month).shift(-1)
    elif rebal =='Quarterly':
        month1 = pd.Series(data.index.quarter)
        month2 = pd.Series(data.index.quarter).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    #data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    transcost = pd.DataFrame(index = data.index)
    transcost['TransCost'] = (trans_cost/10000)*lev_factor
    transcost.index.name='Date'    
    
    rets = pd.DataFrame(columns=['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3'], index=data.index)
    rets.name = 'Momentum Strategy'
    
    for i in range(len(rets)):
        rets['Long1'][i] = str(data.iloc[i,:len(price_data.columns)].idxmax()) + 'M'
        rets['Long2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-1].idxmax())+'M'
        rets['Long3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-2].idxmax())+'M'
        rets['Short1'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[0:].idxmin())+'M'
        rets['Short2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[1:].idxmin())+'M'
        rets['Short3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[2:].idxmin())+'M'
    
    data_new = data.merge(rets[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='Date')
    #new_data.dropna(inplace=True)
    new_data = price_data.join(data_new[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='Date')[rets.index[0]:].ffill()
    
    new_data['Returns'] = new_data.iloc[:,1].copy()*0
    
    if long_short == 'No':
        for i in range(len(new_data)-1):
            if n_securities == 1:
                new_data['Returns'][i+1] = new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1
            elif n_securities == 2:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1))/2
            elif n_securities == 3:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3
    
    if long_short == 'Yes':
        for i in range(len(new_data)-1):
            if n_securities == 1:
                new_data['Returns'][i+1] = (new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1)-(new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) 
            elif n_securities == 2:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1))/2 - ((new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) + 
                                (new_data[new_data['Short2'][i]][i+1] / new_data[new_data['Short2'][i]][i] - 1))/2
            elif n_securities == 3:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3 - ((new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) + 
                                (new_data[new_data['Short2'][i]][i+1] / new_data[new_data['Short2'][i]][i] - 1) + 
                                (new_data[new_data['Short3'][i]][i+1] / new_data[new_data['Short3'][i]][i] - 1))/3
   
    #Merge Value Factor Returns Data with original data and other individual securities returns 
    final_rets = new_data.iloc[:,:len(price_data.columns)].pct_change().dropna().merge(new_data['Returns'], on='Date')
    cols.append('Momentum Factor')
    final_rets.columns = cols
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    final_rets['EW'] = final_rets.iloc[:, :len(final_rets.columns)-1].mean(axis=1)
    
    final_rets = final_rets.merge(cost['EFFR']+spread/(10000*252), on='Date')

    final_rets = (final_rets.drop('EFFR',axis=1) * lev_factor).subtract(final_rets['EFFR']*(lev_factor-1), axis='rows')
    
    final_rets = final_rets.join(transcost['TransCost'], on='Date')
    final_rets['TransCost'] = final_rets['TransCost'].fillna('0')
    final_rets['TransCost'] = final_rets['TransCost'].astype('float64')
    final_rets['Momentum Factor'] = (final_rets['Momentum Factor']).subtract(final_rets['TransCost']*lev_factor, axis='rows')
    final_rets = final_rets.drop('TransCost', axis=1)    
    
    return  final_rets[sample_start:sample_end]    
    
    


def value_factor_backtest(nom_yields, act_yields, real_yields, yield_type, rebal, price_data, zscore_lookback, z_score_smoothening, n_securities, long_short, sample_start, sample_end, cost, spread=0, lev_factor=1, trans_cost=0.00):
    """
    Returns the Monthly Returns of Long Only Value Factor from a given set of bond index returns and real yields by going 
    long on the highest Z-Score Real Yield of the all bond indices every month end
    
    """
    cols = list(price_data.columns)
    #Calculate Z-Score of Real Yields
    if yield_type=='Real Yields - Expected Inflation':
        ry_zscore = (real_yields - real_yields.rolling(260*zscore_lookback).mean())/real_yields.rolling(260*zscore_lookback).std(ddof=1)
        ry_zscore = ry_zscore.dropna().rolling(z_score_smoothening).mean()
        ry_zscore.columns = price_data.columns
    elif yield_type=='Real Yields - Actual Inflation':
        ry_zscore = (act_yields - act_yields.rolling(260*zscore_lookback).mean())/act_yields.rolling(260*zscore_lookback).std(ddof=1)
        ry_zscore = ry_zscore.dropna().rolling(z_score_smoothening).mean()
        ry_zscore.columns = price_data.columns
    elif yield_type=='Nominal Yields':
        ry_zscore = (nom_yields - nom_yields.rolling(260*zscore_lookback).mean())/nom_yields.rolling(260*zscore_lookback).std(ddof=1)
        ry_zscore = ry_zscore.dropna().rolling(z_score_smoothening).mean()
        ry_zscore.columns = price_data.columns
    
    price_data.columns = price_data.columns + str('M')
    
    #Merge Z-Score & Total Return Indices Data
    data = ry_zscore.merge(price_data, on='Date').dropna()
    
    #Convert Data Frequency to Rebalancing Frequency
    #data = data1.asfreq(""+str(rebalancing_period)+"D")
    if rebal =='Monthly':
        month1 = pd.Series(data.index.month)
        month2 = pd.Series(data.index.month).shift(-1)
    elif rebal =='Quarterly':
        month1 = pd.Series(data.index.quarter)
        month2 = pd.Series(data.index.quarter).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    #data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    transcost = pd.DataFrame(index = data.index)
    transcost['TransCost'] = (trans_cost/10000)*lev_factor
    transcost.index.name='Date'    
    
    #Rename Columns for better understanding
    
    rets = pd.DataFrame(columns=['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3'], index=data.index)
    rets.name = 'Value Strategy'
    
    for i in range(len(rets)):
        rets['Long1'][i] = str(data.iloc[i,:len(price_data.columns)].idxmax()) + 'M'
        rets['Long2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-1].idxmax())+'M'
        rets['Long3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-2].idxmax())+'M'
        rets['Short1'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[0:].idxmin())+'M'
        rets['Short2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[1:].idxmin())+'M'
        rets['Short3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[2:].idxmin())+'M'
    
    data_new = data.merge(rets[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='Date')
    #new_data.dropna(inplace=True)
    new_data = price_data.join(data_new[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='Date')[rets.index[0]:].ffill()
    
    new_data['Returns'] = new_data.iloc[:,1].copy()*0
    
    if long_short == 'No':
        for i in range(len(new_data)-1):
            if n_securities == 1:
                new_data['Returns'][i+1] = new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1
            elif n_securities == 2:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1))/2
            elif n_securities == 3:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3
    
    if long_short == 'Yes':
        for i in range(len(new_data)-1):
            if n_securities == 1:
                new_data['Returns'][i+1] = (new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1)-(new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) 
            elif n_securities == 2:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1))/2 - ((new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) + 
                                (new_data[new_data['Short2'][i]][i+1] / new_data[new_data['Short2'][i]][i] - 1))/2
            elif n_securities == 3:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3 - ((new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) + 
                                (new_data[new_data['Short2'][i]][i+1] / new_data[new_data['Short2'][i]][i] - 1) + 
                                (new_data[new_data['Short3'][i]][i+1] / new_data[new_data['Short3'][i]][i] - 1))/3
   
    #Merge Value Factor Returns Data with original data and other individual securities returns 
    final_rets = new_data.iloc[:,:len(price_data.columns)].pct_change().dropna().merge(new_data['Returns'], on='Date')
    cols.append('Value Factor')
    final_rets.columns = cols
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    final_rets['EW'] = final_rets.iloc[:, :len(final_rets.columns)-1].mean(axis=1)
    
    final_rets = final_rets.merge(cost['EFFR']+spread/(10000*252), on='Date')

    final_rets = (final_rets.drop('EFFR',axis=1) * lev_factor).subtract(final_rets['EFFR']*(lev_factor-1), axis='rows')
    
    final_rets = final_rets.join(transcost['TransCost'], on='Date')
    final_rets['TransCost'] = final_rets['TransCost'].fillna('0')
    final_rets['TransCost'] = final_rets['TransCost'].astype('float64')
    final_rets['Value Factor'] = (final_rets['Value Factor']).subtract(final_rets['TransCost']*lev_factor, axis='rows')
    final_rets = final_rets.drop('TransCost', axis=1)
    
    return final_rets[sample_start:sample_end]


#Trend Filtered Value Factor

def trend_value_factor_backtest(real_yields, price_data, zscore_lookback, lookback_period, z_score_smoothening, sample_start, sample_end):
    """
    Returns the Monthly Returns of Long Only Value Factor from a given set of bond index returns and real yields by going 
    long on the highest Z-Score Real Yield of the all bond indices every month end
    
    """
    cols = list(price_data.columns)
    #Calculate Z-Score of Real Yields
    ry_zscore = (real_yields - real_yields.rolling(260*zscore_lookback).mean())/real_yields.rolling(260*zscore_lookback).std(ddof=1)
    ry_zscore = ry_zscore.dropna().rolling(z_score_smoothening).mean()
    ry_zscore.columns = price_data.columns + str('Z')
    
    mom = price_data.pct_change(lookback_period).dropna()
    
    mom.columns = price_data.columns + str('M')
    
    #Merge Z-Score & Total Return Indices Data
    data = price_data.merge(ry_zscore, on='DATE').merge(mom, on='DATE').dropna()
    
    #Convert Data Frequency to Rebalancing Frequency
    month1 = pd.Series(data.index.month)
    month2 = pd.Series(data.index.month).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    #Rename Columns for better understanding
    
    rets = pd.DataFrame(columns=['Long1', 'Long2', 'Long3'], index=data.index)
    rets.name = 'Trend Value Strategy'
    
    filter1 = pd.Series(data='NaN', index=data.index)
    
    for i in range(len(rets)):
        filter1[i] = pd.Series(list(data.iloc[i,len(price_data.columns)*2:][data.iloc[i,len(price_data.columns)*2:]>0].index)).str[:-1]
        rets['Long1'][i] = str(data.iloc[i,:len(price_data.columns)][filter1[i]].idxmax())
        rets['Long2'][i] = str(data.iloc[i,:len(price_data.columns)][filter1[i]].sort_values()[:len(price_data.columns)-1].idxmax())
        rets['Long3'][i] = str(data.iloc[i,:len(price_data.columns)][filter1[i]].sort_values()[:len(price_data.columns)-2].idxmax())

    
    data_new = data.merge(rets[['Long1', 'Long2', 'Long3']], on='DATE')
    #new_data.dropna(inplace=True)
    new_data = price_data.join(data_new[['Long1', 'Long2', 'Long3']], on='DATE')[rets.index[0]:].ffill()
    
    new_data['Returns'] = new_data.iloc[:,1].copy()*0
    

    for i in range(len(new_data)-1):
        new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3
    
   
    #Merge Value Factor Returns Data with original data and other individual securities returns 
    final_rets = new_data.iloc[:,:len(price_data.columns)].pct_change().dropna().merge(new_data['Returns'], on='DATE')
    cols.append('Trend Value Factor')
    final_rets.columns = cols
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    final_rets['EW'] = final_rets.iloc[:, :len(final_rets.columns)-1].mean(axis=1)
    
    return final_rets[:sample_end]

#Low Volatility Factor

def volatility_factor_backtest(nom_yields, act_yields, real_yields, yield_type, rebal, price_data, lookback_period, n_securities, long_short, sample_start, sample_end, cost, spread=0, lev_factor=1, trans_cost=0.00):
    """
    Backtest Momentum Factor based on month end rebalancing and lookback period provided.
    Capaable of long/short and long only daily portfolio returns choosing top 1 to top 3 and bottom 1 to bottom 3 securities
    """
    cols = list(price_data.columns)
    if yield_type=='Real Yields - Expected Inflation':
        mom = real_yields.pct_change().dropna().rolling(lookback_period).std().dropna()/[1,2,4,8,25,10,10,8,3,2,5,24,10,5,4,9,6,6,6,4,8,4]
    elif yield_type=='Real Yields - Actual Inflation':
        mom = act_yields.pct_change().dropna().rolling(lookback_period).std().dropna()/[1,2,4,8,25,10,10,8,3,2,5,24,10,5,4,9,6,6,6,4,8,4]
    elif yield_type=='Nominal Yields':
        mom = nom_yields.pct_change().dropna().rolling(lookback_period).std().dropna()/[1,2,4,8,25,10,10,8,3,2,5,24,10,5,4,9,6,6,6,4,8,4]
        #[2,4,8,4,4,8,2,4,8,24,4]
        
    mom.columns= price_data.columns
    prices = price_data
    prices.columns = price_data.columns + str('M')
    data = mom.merge(prices, on='Date')
    
    #Convert Data Frequency to Rebalancing Frequency
    #data = data.asfreq(""+str(rebalancing_period)+"D")
    if rebal =='Monthly':
        month1 = pd.Series(data.index.month)
        month2 = pd.Series(data.index.month).shift(-1)
    elif rebal =='Quarterly':
        month1 = pd.Series(data.index.quarter)
        month2 = pd.Series(data.index.quarter).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    #data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    transcost = pd.DataFrame(index = data.index)
    transcost['TransCost'] = (trans_cost/10000)*lev_factor
    transcost.index.name='Date'
    
    
    rets = pd.DataFrame(columns=['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3'], index=data.index)
    rets.name = 'Low Volatility Strategy'
    
    for i in range(len(rets)):
        rets['Long1'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[0:].idxmin())+'M'
        rets['Long2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[1:].idxmin())+'M'
        rets['Long3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[2:].idxmin())+'M'
        rets['Short1'][i] = str(data.iloc[i,:len(price_data.columns)].idxmax()) + 'M'
        rets['Short2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-1].idxmax())+'M'
        rets['Short3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-2].idxmax())+'M'
    
    data_new = data.merge(rets[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='Date')
    #new_data.dropna(inplace=True)
    new_data = prices.join(data_new[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='Date')[rets.index[0]:].ffill()
    
    new_data['Returns'] = new_data.iloc[:,1].copy()*0
    
    if long_short == 'No':
        for i in range(len(new_data)-1):
            if n_securities == 1:
                new_data['Returns'][i+1] = new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1
            elif n_securities == 2:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1))/2
            elif n_securities == 3:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3
    
    if long_short == 'Yes':
        for i in range(len(new_data)-1):
            if n_securities == 1:
                new_data['Returns'][i+1] = (new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1)-(new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) 
            elif n_securities == 2:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1))/2 - ((new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) + 
                                (new_data[new_data['Short2'][i]][i+1] / new_data[new_data['Short2'][i]][i] - 1))/2
            elif n_securities == 3:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3 - ((new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) + 
                                (new_data[new_data['Short2'][i]][i+1] / new_data[new_data['Short2'][i]][i] - 1) + 
                                (new_data[new_data['Short3'][i]][i+1] / new_data[new_data['Short3'][i]][i] - 1))/3
   
    #Merge Value Factor Returns Data with original data and other individual securities returns 
    final_rets = new_data.iloc[:,:len(price_data.columns)].pct_change().dropna().merge(new_data['Returns'], on='Date')
    cols.append('Low Volatility Factor')
    final_rets.columns = cols
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    final_rets['EW'] = final_rets.iloc[:, :len(final_rets.columns)-1].mean(axis=1)
    
    final_rets = final_rets.merge(cost['EFFR']+spread/(10000*252), on='Date')

    final_rets = (final_rets.drop('EFFR',axis=1) * lev_factor).subtract(final_rets['EFFR']*(lev_factor-1), axis='rows')
    
    final_rets = final_rets.join(transcost['TransCost'], on='Date')
    final_rets['TransCost'] = final_rets['TransCost'].fillna('0')
    final_rets['TransCost'] = final_rets['TransCost'].astype('float64')
    final_rets['Low Volatility Factor'] = (final_rets['Low Volatility Factor']).subtract(final_rets['TransCost']*lev_factor, axis='rows')
    final_rets = final_rets.drop('TransCost', axis=1)
    
    return final_rets[sample_start:sample_end]

#Carry Strategy Backtest

def carry_factor_backtest(real_yields, price_data, smoothening_period, n_securities, long_short, sample_start, sample_end):
    """
    Returns the Monthly Returns of Long Only Value Factor from a given set of bond index returns and real yields by going 
    long on the highest Z-Score Real Yield of the all bond indices every month end
    
    """
    price_data.columns = ['IG 1-3YM', 'IG 3-5YM', 'IG 7-10YM', 'US HYM', 'CrossoverM', 'EM HYM',
       '1-3Y USTM', 'Intermediate USTM', '7-10Y USTM', 'Long Term USTM', 'Euro HYM']
    
    #Real Yields
    real_yields.columns = ['IG 1-3Y', 'IG 3-5Y', 'IG 7-10Y', 'US HY', 'Crossover', 'EM HY',
       '1-3Y UST', 'Intermediate UST', '7-10Y UST', 'Long Term UST', 'Euro HY']
    
    #Merge Real Yields & Total Return Indices Data
    data = real_yields.merge(price_data, on='DATE').dropna()
    
    #Convert Data Frequency to Rebalancing Frequency
    #data = data1.asfreq(""+str(rebalancing_period)+"D")
    month1 = pd.Series(data.index.month)
    month2 = pd.Series(data.index.month).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    #Rename Columns for better understanding
    
    rets = pd.DataFrame(columns=['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3'], index=data.index)
    rets.name = 'Carry Strategy'
    
    for i in range(len(rets)):
        rets['Long1'][i] = str(data.iloc[i,:11].idxmax()) + 'M'
        rets['Long2'][i] = str(data.iloc[i,:11].sort_values()[:10].idxmax())+'M'
        rets['Long3'][i] = str(data.iloc[i,:11].sort_values()[:9].idxmax())+'M'
        rets['Short1'][i] = str(data.iloc[i,:11].sort_values()[0:].idxmin())+'M'
        rets['Short2'][i] = str(data.iloc[i,:11].sort_values()[1:].idxmin())+'M'
        rets['Short3'][i] = str(data.iloc[i,:11].sort_values()[2:].idxmin())+'M'
    
    data_new = data.merge(rets[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='DATE')
    #new_data.dropna(inplace=True)
    new_data = price_data.join(data_new[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='DATE')[rets.index[0]:].ffill()
    
    new_data['Returns'] = new_data['CrossoverM'].copy()*0
    
    if long_short == 'No':
        for i in range(len(new_data)-1):
            if n_securities == 1:
                new_data['Returns'][i+1] = new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1
            elif n_securities == 2:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1))/2
            elif n_securities == 3:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3
    
    if long_short == 'Yes':
        for i in range(len(new_data)-1):
            if n_securities == 1:
                new_data['Returns'][i+1] = (new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1)-(new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) 
            elif n_securities == 2:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1))/2 - ((new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) + 
                                (new_data[new_data['Short2'][i]][i+1] / new_data[new_data['Short2'][i]][i] - 1))/2
            elif n_securities == 3:
                new_data['Returns'][i+1] = ((new_data[new_data['Long1'][i]][i+1] / new_data[new_data['Long1'][i]][i] - 1) + 
                                (new_data[new_data['Long2'][i]][i+1] / new_data[new_data['Long2'][i]][i] - 1) + 
                                (new_data[new_data['Long3'][i]][i+1] / new_data[new_data['Long3'][i]][i] - 1))/3 - ((new_data[new_data['Short1'][i]][i+1] / new_data[new_data['Short1'][i]][i] - 1) + 
                                (new_data[new_data['Short2'][i]][i+1] / new_data[new_data['Short2'][i]][i] - 1) + 
                                (new_data[new_data['Short3'][i]][i+1] / new_data[new_data['Short3'][i]][i] - 1))/3
   
    #Merge Value Factor Returns Data with original data and other individual securities returns 
    final_rets = new_data.iloc[:,:11].pct_change().dropna().merge(new_data['Returns'], on='DATE')
    final_rets.columns = ['IG 1-3', 'IG 3-5', 'IG 7-10', 'US HY',
       'Crossover', 'EM HY', 'UST 1-3', 'UST Int',
       'UST 7-10', 'UST Long', 'Euro HY', 'Carry Strategy']
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    final_rets['EW'] = final_rets[['IG 1-3', 'IG 3-5', 'IG 7-10', 'US HY', 'Crossover', 'EM HY', 'UST 1-3', 'UST Int', 'UST 7-10', 'UST Long','Euro HY']].mean(axis=1)
    
    return final_rets[:sample_end]


def vix_allocator(price_data, lookback_period, sample_start, sample_end):
    
    
    vix = pd.DataFrame(yf.download('^VIX', start='1999-01-01')['Adj Close'])
    vix.index.name='DATE'
    vix.columns = ['VIX']
    month1 = pd.Series(vix.index.month)
    month2 = pd.Series(vix.index.month).shift(-1)
    mask = (month1 != month2)
    data = vix[mask.values]
    data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    data = ((data.rolling(lookback_period).max()).merge(data['VIX'], on='DATE')).dropna()
    data = data['VIX_y']/data['VIX_x']
    
    data1 = pd.DataFrame(data).merge(price_data, on='DATE')
    indices1 = data1.drop(0, axis=1).pct_change().dropna()
    data = pd.DataFrame(data).merge(indices1, on='DATE')
    data.columns = [                  'VIX',          'IG 1-3YM',          'IG 3-5YM',
               'IG 7-10YM',            'US HYM',        'CrossoverM',
                  'EM HYM',         '1-3Y USTM', 'Intermediate USTM',
              '7-10Y USTM',    'Long Term USTM']
    
    data['VIX Switching Strategy'] = data['VIX'].copy()*0
    for i in range(1,len(data)):
        data['VIX Switching Strategy'][i] = (data['VIX'][i-1]*data['Long Term USTM'][i]) + ((1-data['VIX'][i-1]) * data['EM HYM'][i])
    data = data.iloc[1:,:]
    return data.drop('VIX', axis=1)


def fi_vix_allocator(duration, high_yield, vix, lookback_period, vix_smooth_period, sample_start, sample_end):
    """
    Allocates between Duration & Credit Risk using VIX percentile (based on the lookback period provided). Uses the VIX percentile at month end to decide the weight for duration and rest in invested in Credit Risk. Both Duration and Credit Risk Portfolios are equally weighted.
    """

    price_data = duration.merge(high_yield, on='Date')
    cols=list(price_data.columns)
    vix = vix
    vix.index.name='Date'
    vix.columns = ['VIX']
    data = vix
    #data = data[:sample_end]
    data.dropna(inplace=True)
    
    data = ((data.rolling(lookback_period*252).max()).merge(data['VIX'], on='Date')).dropna()
    data = (data['VIX_y']/data['VIX_x']).rolling(vix_smooth_period).mean().dropna()
    
    #data1 = pd.DataFrame(data).merge(price_data, on='DATE')
    data = pd.DataFrame(data).merge(price_data, on='Date')
    cols.insert(0, 'VIX')
    data.columns = cols    
    
    month1 = pd.Series(vix.index.month)
    month2 = pd.Series(vix.index.month).shift(-1)

    
    mask = (month1 != month2)
    mask = mask.astype(int)
    mask.index = vix.index
    mask.name='Rebalance'
    
    indices_ret = price_data.pct_change().dropna()
    strat = price_data.join(data['VIX'], on='Date').join(mask, on='Date').dropna()
    #strat = strat[date(strat.index.year[1], strat.index.month[1], strat.index.day[:30].max()):]
    
    strat['Portfolio'] = strat['VIX']*0+100
    strat['HY Portfolio'] = strat['VIX']*0
    strat['Dur Portfolio'] = strat['VIX']*0
    
    for i in range (1,len(strat['Dur Portfolio'])):
        if strat['Rebalance'][i-1] == 1:
            strat['Dur Portfolio'][i] = strat['Portfolio'][i-1] * strat['VIX'][i] / strat[list(duration.columns)].mean(axis=1)[i-1]
            strat['HY Portfolio'][i] = strat['Portfolio'][i-1] * (1-strat['VIX'][i])/strat[list(high_yield.columns)].mean(axis=1)[i-1]
            strat['Portfolio'][i] = strat['Dur Portfolio'][i] * strat[list(duration.columns)].mean(axis=1)[i] + strat['HY Portfolio'][i] * strat[list(high_yield.columns)].mean(axis=1)[i] 
        elif strat['Rebalance'][i-1] == 0:
            strat['Dur Portfolio'][i] = strat['Dur Portfolio'][i-1]
            strat['HY Portfolio'][i] = strat['HY Portfolio'][i-1]
            strat['Portfolio'][i] = strat['Dur Portfolio'][i] * strat[list(duration.columns)].mean(axis=1)[i] + strat['HY Portfolio'][i] * strat[list(high_yield.columns)].mean(axis=1)[i]
        
    return pd.DataFrame(strat['Portfolio'].pct_change()).merge(price_data.pct_change(), on='Date').dropna()[sample_start :sample_end]
        
        
    
    

#Risk Return Summary Stats

def backtest_stats(returns, rebalancing_period):
    return erk.summary_stats1(returns,0,252/rebalancing_period).sort_values('Total Return').T


#Plot Total Return Charts

def plot_chart(returns, sample_start, sample_end, long_short, factor_name='', vix='No'):
    if long_short == 'Yes':
        returns.iloc[0,:]=0
        df = (1+returns).cumprod()-1
        fig = px.line(df, x=df.index, y=df.columns)
        fig.update_layout(title = 'Fixed Income ' + str(factor_name)+ ' Factor Long/Short In-Sample Backtest',
                       xaxis_title='Date',
                       yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                       legend_title_text='ETFs', plot_bgcolor = 'White', yaxis_tickformat = '%')
        fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}') 
        fig.update_yaxes(automargin=True)
        
        fig.add_annotation(x=0.05, y=0.9, xref="paper", yref="paper", showarrow=False, align='left', text="Annualized Return(CAGR): "+str((erk.annualized_ret(returns.iloc[:,0], 252)*100).round(2)) +"%")
        
        fig.add_annotation(x=0.05, y=0.85, xref="paper", yref="paper",  showarrow=False, align='left', text="Reward/Risk Ratio: "+str(erk.sharpe_ratio(returns.iloc[:,0], 252, 0.0).round(2)))
        #fig.update_xaxes(rangeslider_visible=True)
        fig.show()
    
    else:
        returns.iloc[0,:]=0
        df = (1+returns).cumprod()-1
        fig = px.line(df, x=df.index, y=df.columns)
        if vix=='Yes':
            fig.update_layout(title = 'VIX Based Allocation Strategy In-Sample Backtest',
                       xaxis_title='Date',
                       yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                       legend_title_text='ETFs', plot_bgcolor = 'White', yaxis_tickformat = '%')
            fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}') 
            fig.update_yaxes(automargin=True)
        else: 
            fig.update_layout(title = 'Fixed Income ' + str(factor_name)+ ' Factor Long Only In-Sample Backtest',
                           xaxis_title='Date',
                           yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                           legend_title_text='ETFs', plot_bgcolor = 'White', yaxis_tickformat = '%')
            fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}') 
            fig.update_yaxes(automargin=True)
        
        fig.add_annotation(x=0.05, y=0.9, xref="paper", yref="paper", showarrow=False, align='left', text="Annualized Return(CAGR): "+str((erk.annualized_ret(returns.iloc[:,0], 252)*100).round(2)) +"%")
        
        fig.add_annotation(x=0.05, y=0.85, xref="paper", yref="paper",  showarrow=False, align='left', text="Reward/Risk Ratio: "+str(erk.sharpe_ratio(returns.iloc[:,0], 252, 0.0).round(2)))
        #fig.update_xaxes(rangeslider_visible=True)
        fig.show()

        
def import_data(yields_d='Expected_Infl'):
    #Expected Inflation Data from Cleveland Fed & Yields
    infl = pd.read_excel('expected_inflation_rates.xlsx', index_col=0, header=0, parse_dates=True)
    infl.index.name = 'DATE'
    
    yields = pd.read_excel('fi_yields.xlsx', header=0, index_col=0, parse_dates=True)
    yields.dropna(inplace=True)
    yields.index.name = 'DATE'
    
    cpi = pd.read_excel('cpi-edit.xlsx', header=0, index_col=0, parse_dates=True)
    cpi.index.name='DATE'
    cpi = cpi.ffill()
    
    #data = infl.merge(yields, on='DATE')
    #Price Data
    indices = pd.read_excel('fi_prices.xlsx', header=0, index_col=0, parse_dates=True)
    indices.dropna(inplace=True)
    #indices.plot(figsize=(12,6), fontsize=13, title='TR Indices')
    
    if yields_d=='Expected_Infl':
        data = infl.merge(yields, on='DATE')
        data.dropna(inplace=True)
        #Real Yields
        ig1_3ry = pd.DataFrame(data['IG 1-3Y Yield'] - data[' 2 year Expected Inflation'])
        ig3_5ry = pd.DataFrame(data['IG 3-5Y Yield'] - data[' 4 year Expected Inflation'])
        ig7_10ry = pd.DataFrame(data['IG 7-10Y Yield'] - data[' 8 year Expected Inflation'])
        us_hyry = pd.DataFrame(data['US HYield'] - data[' 4 year Expected Inflation'])
        crossry = pd.DataFrame(data['Crossover Yield'] - data[' 4 year Expected Inflation'])
        em_hyry = pd.DataFrame(data['EM HY Yield'] - data[' 8 year Expected Inflation'])
        ust1_3ry = pd.DataFrame(data['UST 1-3Y Yield'] - data[' 2 year Expected Inflation'])
        ust_intry = pd.DataFrame(data['UST Int Yield'] - data[' 4 year Expected Inflation'])
        ust7_10ry = pd.DataFrame(data['UST 7-10Y Yield'] - data[' 8 year Expected Inflation'])
        ust_longry = pd.DataFrame(data['UST Long  Yield'] - data[' 24 year Expected Inflation'])
        euro_hyry = pd.DataFrame(data['Euro HY Yield'] - data[' 4 year Expected Inflation'])

        real_yields = ig1_3ry.merge(ig3_5ry, on='DATE').merge(ig7_10ry, on='DATE').merge(us_hyry, on='DATE').merge(crossry, on='DATE').merge(em_hyry, on='DATE').merge(ust1_3ry, on='DATE').merge(ust_intry, on='DATE').merge(ust7_10ry, on='DATE').merge(ust_longry, on='DATE').merge(euro_hyry, on='DATE')
        real_yields.dropna(inplace=True)
        real_yields.columns = ['IG 1-3 Yield', 'IG 3-5 Yield', 'IG 7-10 Yield', 'US HY Yield', 'Crossover Yield', 'EM High Yield', 'UST 1-3 Yield', 'UST Int Yield', 'UST 7-10 Yield', 'UST Long Yield', 'Euro High Yield']
        
    elif yields_d=='Actual_Infl':
        data = cpi.merge(yields, on='DATE')
        data.dropna(inplace=True)
        #Real Yields
        ig1_3ry = pd.DataFrame(data['IG 1-3Y Yield'] - data['2Y'])
        ig3_5ry = pd.DataFrame(data['IG 3-5Y Yield'] - data['4Y'])
        ig7_10ry = pd.DataFrame(data['IG 7-10Y Yield'] - data['8Y'])
        us_hyry = pd.DataFrame(data['US HYield'] - data['4Y'])
        crossry = pd.DataFrame(data['Crossover Yield'] - data['4Y'])
        em_hyry = pd.DataFrame(data['EM HY Yield'] - data['8Y'])
        ust1_3ry = pd.DataFrame(data['UST 1-3Y Yield'] - data['2Y'])
        ust_intry = pd.DataFrame(data['UST Int Yield'] - data['4Y'])
        ust7_10ry = pd.DataFrame(data['UST 7-10Y Yield'] - data['8Y'])
        ust_longry = pd.DataFrame(data['UST Long  Yield'] - data['24Y'])
        euro_hyry = pd.DataFrame(data['Euro HY Yield'] - data['4Y'])

        real_yields = ig1_3ry.merge(ig3_5ry, on='DATE').merge(ig7_10ry, on='DATE').merge(us_hyry, on='DATE').merge(crossry, on='DATE').merge(em_hyry, on='DATE').merge(ust1_3ry, on='DATE').merge(ust_intry, on='DATE').merge(ust7_10ry, on='DATE').merge(ust_longry, on='DATE').merge(euro_hyry, on='DATE')
        real_yields.dropna(inplace=True)
        real_yields.columns = ['IG 1-3 Yield', 'IG 3-5 Yield', 'IG 7-10 Yield', 'US HY Yield', 'Crossover Yield', 'EM High Yield', 'UST 1-3 Yield', 'UST Int Yield', 'UST 7-10 Yield', 'UST Long Yield', 'Euro High Yield']
            
        
    yields.columns = ['IG 1-3 Yield', 'IG 3-5 Yield', 'IG 7-10 Yield', 'US HY Yield', 'Crossover Yield', 'EM High Yield', 'UST 1-3 Yield', 'UST Int Yield', 'UST 7-10 Yield', 'UST Long Yield', 'Euro High Yield']
    
    if yields_d=='Expected_Infl' or yields_d=='Actual_Infl' :
        #real_yields.plot(figsize=(15,7), title='Real Yields')
        return real_yields
    if yields_d=='Nominal':
        #yields.plot(figsize=(15,7), title='Nominal Yields')
        return yields
    
def import_yield_data(yields_d='Expected_Infl'):
    #Expected Inflation Data from Cleveland Fed & Yields
    indices = pd.read_excel('Total Return Index - FI.xlsx', header=0, index_col=0, parse_dates=True)
    indices.dropna(inplace=True)
    nom_yields = pd.read_excel('Nominal Yields - FI.xlsx', header=0, index_col=0, parse_dates=True)
    nom_yields.dropna(inplace=True)
    
    act_infl = pd.read_excel('Actual Inflation - FI.xlsx', header=0, index_col=0, parse_dates=True)
    exp_infl = pd.read_excel('Expected Inflation - FI.xlsx', header=0, index_col=0, parse_dates=True)
    act_infl = act_infl.ffill()
    exp_infl = exp_infl.ffill()
    exp_infl.index.name = 'Date'

    
    
    if yields_d=='Expected_Infl':
        real_yields = nom_yields.merge(exp_infl, on='Date')
        
        real_yields['US 1-3Y Treasuries'] = real_yields['US 1-3Y Treasuries'] - real_yields['2Y']
        real_yields['US Int Treasuries'] = real_yields['US Int Treasuries'] - real_yields['4Y']
        real_yields['US 7-10Y Treasuries'] = real_yields['US 7-10Y Treasuries'] - real_yields['8Y']
        real_yields['US Long Treasuries'] = real_yields['US Long Treasuries'] - real_yields['25Y']
        real_yields['Intl Sovereign Bonds'] = real_yields['Intl Sovereign Bonds'] - real_yields['10Y']
        real_yields['Global Treasuries'] = real_yields['Global Treasuries'] - real_yields['10Y']
        real_yields['US TIPS'] = real_yields['US TIPS'] - real_yields['8Y']
        real_yields['US IG 1-5Y'] = real_yields['US IG 1-5Y'] - real_yields['3Y']
        real_yields['US IG 1-3Y'] = real_yields['US IG 1-3Y'] - real_yields['2Y']
        real_yields['US IG Int'] = real_yields['US IG Int'] - real_yields['5Y']
        real_yields['US IG Long'] = real_yields['US IG Long'] - real_yields['24Y']
        real_yields['Global IG'] = real_yields['Global IG'] - real_yields['10Y']
        real_yields['Developed Mkts IG'] = real_yields['Developed Mkts IG'] - real_yields['5Y']
        real_yields['US MBS'] = real_yields['US MBS'] - real_yields['4Y']
        real_yields['Global Aggregate'] = real_yields['Global Aggregate'] - real_yields['9Y']
        real_yields['Euro Aggregate'] = real_yields['Euro Aggregate'] - real_yields['6Y']
        real_yields['US HY'] = real_yields['US HY'] - real_yields['6Y']
        real_yields['US HY Beta'] = real_yields['US HY Beta'] - real_yields['6Y']
        real_yields['US Crossover'] = real_yields['US Crossover'] - real_yields['4Y']
        real_yields['EM HY'] = real_yields['EM HY'] - real_yields['8Y']
        real_yields['Euro HY'] = real_yields['Euro HY'] - real_yields['4Y']
        
        real_yields = real_yields[list(nom_yields.columns)]
        
    elif yields_d=='Actual_Infl':
        act_yields = nom_yields.merge(act_infl, on='Date')
        
        act_yields['US 1-3Y Treasuries'] = act_yields['US 1-3Y Treasuries'] - act_yields['2Y']
        act_yields['US Int Treasuries'] = act_yields['US Int Treasuries'] - act_yields['4Y']
        act_yields['US 7-10Y Treasuries'] = act_yields['US 7-10Y Treasuries'] - act_yields['8Y']
        act_yields['US Long Treasuries'] = act_yields['US Long Treasuries'] - act_yields['24Y']
        act_yields['Intl Sovereign Bonds'] = act_yields['Intl Sovereign Bonds'] - act_yields['10Y']
        act_yields['Global Treasuries'] = act_yields['Global Treasuries'] - act_yields['10Y']
        act_yields['US TIPS'] = act_yields['US TIPS'] - act_yields['8Y']
        act_yields['US IG 1-5Y'] = act_yields['US IG 1-5Y'] - act_yields['3Y']
        act_yields['US IG 1-3Y'] = act_yields['US IG 1-3Y'] - act_yields['2Y']
        act_yields['US IG Int'] = act_yields['US IG Int'] - act_yields['5Y']
        act_yields['US IG Long'] = act_yields['US IG Long'] - act_yields['24Y']
        act_yields['Global IG'] = act_yields['Global IG'] - act_yields['10Y']
        act_yields['Developed Mkts IG'] = act_yields['Developed Mkts IG'] - act_yields['5Y']
        act_yields['US MBS'] = act_yields['US MBS'] - act_yields['4Y']
        act_yields['Global Aggregate'] = act_yields['Global Aggregate'] - act_yields['8Y']
        act_yields['Euro Aggregate'] = act_yields['Euro Aggregate'] - act_yields['6Y']
        act_yields['US HY'] = act_yields['US HY'] - act_yields['6Y']
        act_yields['US HY Beta'] = act_yields['US HY Beta'] - act_yields['6Y']
        act_yields['US Crossover'] = act_yields['US Crossover'] - act_yields['4Y']
        act_yields['EM HY'] = act_yields['EM HY'] - act_yields['8Y']
        act_yields['Euro HY'] = act_yields['Euro HY'] - act_yields['4Y']
        
        act_yields = act_yields[list(nom_yields.columns)]
        
                
    
    if yields_d=='Expected_Infl':
        #real_yields.plot(figsize=(15,7), title='Real Yields')
        return real_yields
    elif yields_d=='Actual_Infl':
        return act_yields
    elif yields_d=='Nominal':
        #yields.plot(figsize=(15,7), title='Nominal Yields')
        return nom_yields