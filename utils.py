import statsmodels.tsa as tsa
from statsmodels.tsa.vector_ar.var_model import VAR, FEVD
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, ccf, ccovf, kpss
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pandas as pd

# Testing for stationarity

def adf_test(series):
    result = adfuller(series)
    print('Augmented Dicky Fuller Test')
    labels = ['Test Statistic','p-value','Lag Used']
    output = list(result[:3]) + [result[-2]['5%'], result[-2]['1%']]
    # for value, label in zip(output,labels):
    #     print(label+': '+str(round(value, 7)))
    print(f'\tResult: The series is {"" if result[1] <= .05 else "non-"}stationary')
        
def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    print('KPSS Test')
    # Format Output
    # print(f'KPSS Statistic: {statistic}')
    # print(f'p-value: {p_value}')
    # print(f'num lags: {n_lags}')
    # print('Critial Values:')
    # for key, value in critical_values.items():
    #     print(f'   {key} : {value}')
    print(f'\tResult: The series is {"non-" if p_value <= 0.05 else ""}stationary')
# Granger Causality Test

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False,maxlag=12):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
def adjust(val, length= 6): return str(val).ljust(length)
def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    # Summary
    print(f'{"Name":31}{"::":4}{"Test Stat":10}{">":2}{"C(95%)":10}{"=>":4}Signif', '\n','--'*35)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col,30), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
# Train-Test Split
def splitter(data_df):
    end = round(len(data_df)*.8)
    train_df = data_df[:end]
    test_df = data_df[end:]
    return train_df, test_df
def create_plot(data1, data2, crypto, model):
    fig, ax1 = plt.subplots()
    color = 'blue'
    ax1.set_xlabel('date')
    ax1.set_ylabel('price', color='black')
    ax1.plot(data2.index, data2, color=color, label=f'Actual {crypto} Price')
    ax1.tick_params(axis='y', labelcolor='black')


    color = 'orange'
    # ax2.set_ylabel('Price', color='black')  # we already handled the x-label with ax1
    ax1.plot(data1.index, data1, color=color, label=f'Forecasted {crypto} Price')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.tick_params(axis='x', labelrotation=90)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 , labels1 , loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=3)
    ax1.set_xlabel('Date', color='black') 

    plt.title(f'Forecast vs Actuals for {crypto} price (USD) ({model}))')
    # plt.xticks(ts_pos_sub.index)
    plt.grid()
    plt.show()
def forecast_accuracy(forecast, actual, rowname):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))*100  # MAPE
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mse = np.mean((forecast - actual)**2)         # MSE
    rmse = np.mean((forecast - actual)**2)**.5 
    metrics_df = pd.DataFrame({'MAE': [mae],  'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse]},index = rowname)
    return metrics_df