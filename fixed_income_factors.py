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

### Momentum Factor Backtest

def momentum_factor_backtest_monthly(price_data, lookback_period, n_securities, long_short, sample_start, sample_end):
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
    month1 = pd.Series(data.index.month)
    month2 = pd.Series(data.index.month).shift(-1)
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






def momentum_factor_backtest(price_data, lookback_period, duration_adj, n_securities, long_short, sample_start, sample_end):
    """
    Backtest Momentum Factor based on month end rebalancing and lookback period provided. Factor is adjusted for duration by dividing lookback percentage return with weighted average duration of fund/index.
    Capable of long/short and long only daily portfolio returns choosing top 1 to top 3 and bottom 1 to bottom 3 securities
    """
    if duration_adj== 'Yes':
        mom = price_data.pct_change(lookback_period).dropna()/[2,4,8,4,4,8,2,4,8,24,4]
    else:
        mom = price_data.pct_change(lookback_period).dropna()
    
    cols = list(price_data.columns)
    mom.columns= price_data.columns
    price_data.columns = price_data.columns + str('M')

    data = mom.merge(price_data, on='DATE')
    
    #Convert Data Frequency to Rebalancing Frequency
    #data = data.asfreq(""+str(rebalancing_period)+"D")
    month1 = pd.Series(data.index.month)
    month2 = pd.Series(data.index.month).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    rets = pd.DataFrame(columns=['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3'], index=data.index)
    rets.name = 'Momentum Strategy'
    
    for i in range(len(rets)):
        rets['Long1'][i] = str(data.iloc[i,:len(price_data.columns)].idxmax()) + 'M'
        rets['Long2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-1].idxmax())+'M'
        rets['Long3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-2].idxmax())+'M'
        rets['Short1'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[0:].idxmin())+'M'
        rets['Short2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[1:].idxmin())+'M'
        rets['Short3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[2:].idxmin())+'M'
    
    data_new = data.merge(rets[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='DATE')
    #new_data.dropna(inplace=True)
    new_data = price_data.join(data_new[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='DATE')[rets.index[0]:].ffill()
    
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
    final_rets = new_data.iloc[:,:len(price_data.columns)].pct_change().dropna().merge(new_data['Returns'], on='DATE')
    cols.append('Momentum Factor')
    final_rets.columns = cols
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    final_rets['EW'] = final_rets.iloc[:, :len(final_rets.columns)-1].mean(axis=1)
    
    return final_rets[:sample_end]


def value_factor_backtest(real_yields, price_data, zscore_lookback, z_score_smoothening, n_securities, long_short, sample_start, sample_end):
    """
    Returns the Monthly Returns of Long Only Value Factor from a given set of bond index returns and real yields by going 
    long on the highest Z-Score Real Yield of the all bond indices every month end
    
    """
    cols = list(price_data.columns)
    #Calculate Z-Score of Real Yields
    ry_zscore = (real_yields - real_yields.rolling(260*zscore_lookback).mean())/real_yields.rolling(260*zscore_lookback).std(ddof=1)
    ry_zscore = ry_zscore.dropna().rolling(z_score_smoothening).mean()
    ry_zscore.columns = price_data.columns
    
    price_data.columns = price_data.columns + str('M')
    
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
    
    rets = pd.DataFrame(columns=['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3'], index=data.index)
    rets.name = 'Value Strategy'
    
    for i in range(len(rets)):
        rets['Long1'][i] = str(data.iloc[i,:len(price_data.columns)].idxmax()) + 'M'
        rets['Long2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-1].idxmax())+'M'
        rets['Long3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-2].idxmax())+'M'
        rets['Short1'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[0:].idxmin())+'M'
        rets['Short2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[1:].idxmin())+'M'
        rets['Short3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[2:].idxmin())+'M'
    
    data_new = data.merge(rets[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='DATE')
    #new_data.dropna(inplace=True)
    new_data = price_data.join(data_new[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='DATE')[rets.index[0]:].ffill()
    
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
    final_rets = new_data.iloc[:,:len(price_data.columns)].pct_change().dropna().merge(new_data['Returns'], on='DATE')
    cols.append('Value Factor')
    final_rets.columns = cols
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    final_rets['EW'] = final_rets.iloc[:, :len(final_rets.columns)-1].mean(axis=1)
    
    return final_rets[:sample_end]


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

def volatility_factor_backtest(real_yields, price_data, lookback_period, n_securities, long_short, sample_start, sample_end):
    """
    Backtest Momentum Factor based on month end rebalancing and lookback period provided.
    Capaable of long/short and long only daily portfolio returns choosing top 1 to top 3 and bottom 1 to bottom 3 securities
    """
    cols = list(price_data.columns)
    mom = real_yields.pct_change().dropna().rolling(lookback_period).std().dropna()/[2,4,8,4,4,8,2,4,8,24,4]
    mom.columns= price_data.columns
    prices = price_data
    prices.columns = price_data.columns + str('M')
    data = mom.merge(prices, on='DATE')
    
    #Convert Data Frequency to Rebalancing Frequency
    #data = data.asfreq(""+str(rebalancing_period)+"D")
    month1 = pd.Series(data.index.month)
    month2 = pd.Series(data.index.month).shift(-1)
    mask = (month1 != month2)
    data = data[mask.values]
    data = data[sample_start:sample_end]
    data.dropna(inplace=True)
    
    rets = pd.DataFrame(columns=['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3'], index=data.index)
    rets.name = 'Low Volatility Strategy'
    
    for i in range(len(rets)):
        rets['Long1'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[0:].idxmin())+'M'
        rets['Long2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[1:].idxmin())+'M'
        rets['Long3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[2:].idxmin())+'M'
        rets['Short1'][i] = str(data.iloc[i,:len(price_data.columns)].idxmax()) + 'M'
        rets['Short2'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-1].idxmax())+'M'
        rets['Short3'][i] = str(data.iloc[i,:len(price_data.columns)].sort_values()[:len(price_data.columns)-2].idxmax())+'M'
    
    data_new = data.merge(rets[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='DATE')
    #new_data.dropna(inplace=True)
    new_data = prices.join(data_new[['Long1', 'Long2', 'Long3', 'Short1', 'Short2', 'Short3']], on='DATE')[rets.index[0]:].ffill()
    
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
    final_rets = new_data.iloc[:,:len(price_data.columns)].pct_change().dropna().merge(new_data['Returns'], on='DATE')
    cols.append('Low Volatility Factor')
    final_rets.columns = cols
    
    #Add Equally Weighted Portfolio Returns for comparison as well
    final_rets['EW'] = final_rets.iloc[:, :len(final_rets.columns)-1].mean(axis=1)
    
    return final_rets[:sample_end]

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


def fi_vix_allocator(duration, high_yield, lookback_period, vix_smooth_period, sample_end):
    """
    Allocates between Duration & Credit Risk using VIX percentile (based on the lookback period provided). Uses the VIX percentile at month end to decide the weight for duration and rest in invested in Credit Risk. Both Duration and Credit Risk Portfolios are equally weighted.
    """

    price_data = duration.merge(high_yield, on='DATE')
    cols=list(price_data.columns)
    vix = pd.DataFrame(yf.download('^VIX', start='1999-01-01')['Adj Close'])
    vix.index.name='DATE'
    vix.columns = ['VIX']
    data = vix
    #data = data[:sample_end]
    data.dropna(inplace=True)
    
    data = ((data.rolling(lookback_period*252).max()).merge(data['VIX'], on='DATE')).dropna()
    data = (data['VIX_y']/data['VIX_x']).rolling(vix_smooth_period).mean().dropna()
    
    #data1 = pd.DataFrame(data).merge(price_data, on='DATE')
    data = pd.DataFrame(data).merge(price_data, on='DATE')
    cols.insert(0, 'VIX')
    data.columns = cols    
    
    month1 = pd.Series(vix.index.month)
    month2 = pd.Series(vix.index.month).shift(-1)
    mask = (month1 != month2)
    mask = mask.astype(int)
    mask.index = vix.index
    mask.name='Rebalance'
    
    indices_ret = price_data.pct_change().dropna()
    strat = price_data.join(data['VIX'], on='DATE').join(mask, on='DATE').dropna()[:sample_end]
    strat = strat[date(strat.index.year[1], strat.index.month[1], strat.index.day[:30].max()):]
    
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
        
    return pd.DataFrame(strat['Portfolio'].pct_change()).merge(price_data.pct_change(), on='DATE').dropna()
        
        
    
    

#Risk Return Summary Stats

def backtest_stats(returns, rebalancing_period):
    return erk.summary_stats1(returns,0,252/rebalancing_period).sort_values('Total Return').T


#Plot Total Return Charts

def plot_chart(returns, sample_start, sample_end, long_short, factor_name=''):
    if long_short == 'Yes':
        b = (1+returns.iloc[:,-2]).cumprod().plot(figsize=(13,8), fontsize=13)
        plt.title('Fixed Income ' + str(factor_name)+ ' Factor Long Short In-Sample Backtest', fontsize=14)
        plt.legend(loc=(1,0.1))
        b.annotate("Annualized Return(CAGR): "+str((erk.annualized_ret(returns.iloc[:,-2], 252)*100).round(2)) +"%", xy=(.05, .90), xycoords='axes fraction', fontsize=14)
        b.annotate("Sharpe Ratio: "+str(erk.sharpe_ratio(returns.iloc[:,-2], 252, 0.0).round(2)), xy=(.05, .85), xycoords='axes fraction', fontsize=14)
    else:
        b = (1+returns).cumprod().plot(figsize=(13,8), fontsize=13)
        plt.title('Fixed Income ' + str(factor_name)+ ' Factor Long Only In-Sample Backtest', fontsize=14)
        plt.legend(loc=(1,0.1))
        b.annotate("Annualized Return(CAGR): "+str((erk.annualized_ret(returns.iloc[:,-2], 252)*100).round(2)) +"%", xy=(.05, .90), xycoords='axes fraction', fontsize=14)
        b.annotate("Sharpe Ratio: "+str(erk.sharpe_ratio(returns.iloc[:,-2], 252, 0.0).round(2)), xy=(.05, .85), xycoords='axes fraction', fontsize=14)
        
def import_data():
    #Expected Inflation Data from Cleveland Fed & Yields
    infl = pd.read_excel('expected_inflation_rates.xlsx', index_col=0, header=0, parse_dates=True)
    infl.index.name = 'DATE'
    yields = pd.read_excel('fi_yields.xlsx', header=0, index_col=0, parse_dates=True)
    yields.dropna(inplace=True)
    yields.index.name = 'DATE'
    data = infl.merge(yields, on='DATE')
    #Price Data
    indices = pd.read_excel('fi_prices.xlsx', header=0, index_col=0, parse_dates=True)
    indices.dropna(inplace=True)
    indices.plot(figsize=(12,6), fontsize=13, title='TR Indices')
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
    real_yields.plot(figsize=(15,7), title='Real Yields')
    
    return real_yields