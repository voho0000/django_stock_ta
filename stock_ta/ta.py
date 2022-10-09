import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import talib
import pandas_ta as ta
from scipy.signal import argrelextrema
import numpy as np

def get_raw_df(code, period, interval):
    raw_df = yf.download(str(code)+'.tw', period=str(period), interval=str(interval))
    return raw_df

def get_ta_df(raw_df, willr=9, bias=5, rsi=6):
    k_9, d_9 = talib.STOCH(raw_df['High'],raw_df['Low'],raw_df['Close'], 
                       fastk_period=9,
                       slowk_period=3,
                       slowk_matype=1,
                       slowd_period=3,
                       slowd_matype=1)
    upper, middle, lower = talib.BBANDS(raw_df["Adj Close"], timeperiod=5)
    ta_df = pd.DataFrame(index=raw_df.index,
                      data = {
                          "bias": raw_df.ta.bias(length=bias),
                          "bb_low": lower,
                          "bb_ma": middle,
                          "bb_high": upper,
                          "k_9":k_9,
                          "d_9":d_9,
                          "k-d_9":k_9-d_9,
                          "RSI":talib.RSI(raw_df['Close'], timeperiod=rsi),
                          "WILLR":talib.WILLR(raw_df['High'],raw_df['Low'],raw_df['Close'], timeperiod=willr)
                      }
                        )
    return ta_df

def check_empty(raw_df, willr, bias, rsi):
    k = 9
    max_index = np.argmax([k, willr, bias, rsi])
    if len(raw_df)<max([k, willr, bias, rsi]):
        return True
    else:
        ta_df = get_ta_df(raw_df, willr, bias, rsi)
        if (ta_df.sum(axis=0)['k_9']==0) | (ta_df.sum(axis=0)[['k_9','WILLR', 'bias', 'RSI'][max_index]]==0):
            return True
        else:
            return False

def TopDivergence(ta_df, raw_df, num=3):
    # num: check the number of the closest peaks
    signal = []
    TopSignal = {'time':[],'value':[]}
    localMax = argrelextrema(np.array(ta_df), np.greater)
    localMax = list(localMax[0])
    if len(localMax)<num:
        num = len(localMax)
    for i,value in enumerate(ta_df):
        if i in localMax:
            signal.append(value+0.002)
        else:
            signal.append(np.nan)
    if not localMax:
        return TopSignal
    else:
        for n in range(1, num+1):
            current_ind = localMax[-n]
            greater=[i for i in signal[:-n] if i>signal[current_ind] ]   
            if not greater:
                continue
            else:
                greater_ind = signal.index(greater[-1])
                if raw_df['Close'][current_ind] > raw_df['Close'][greater_ind]:
                    TopSignal['time'].append([raw_df.index[current_ind], raw_df.index[greater_ind]])
                    for i,value in enumerate(ta_df):
                        if i in [current_ind, greater_ind]:
                            TopSignal['value'].append(value)
                        else:
                            TopSignal['value'].append(np.nan)
                    return TopSignal
                    break
                else:
                    continue
        if not TopSignal['value']:
            return TopSignal

def BottomDivergence(ta_df, raw_df, num=3):
    # num: check the number of the closest peaks
    signal = []
    BottomSignal = {'time':[],'value':[]}
    localMin = argrelextrema(np.array(ta_df), np.less)
    localMin = list(localMin[0])
    if len(localMin)<num:
        num = len(localMin)
    for i,value in enumerate(ta_df):
        if i in localMin:
            signal.append(value-0.002)
        else:
            signal.append(np.nan)
    if not localMin:
        return BottomSignal
    else:
        for n in range(1, num+1):
            current_ind = localMin[-n]
            lesser=[i for i in signal[:-n] if i<signal[current_ind] ]  
            if not lesser:
                continue
            else:
                lesser_ind = signal.index(lesser[-1])
                if raw_df['Close'][current_ind] > raw_df['Close'][lesser_ind]:
                    BottomSignal['time'].append([raw_df.index[current_ind], raw_df.index[lesser_ind]])
                    for i,value in enumerate(ta_df):
                        if i in [current_ind, lesser_ind]:
                            BottomSignal['value'].append(value)
                        else:
                            BottomSignal['value'].append(np.nan)
                    return BottomSignal
                    break
                else:
                    continue
        if not BottomSignal['value']:
            return BottomSignal

def get_Divergences(ta_df, raw_df):
    raw_df.index = raw_df.index.strftime('%Y-%m-%d %H:%M')
    bias_Top =TopDivergence(ta_df["bias"], raw_df)
    k_9_Top =TopDivergence(ta_df["k_9"], raw_df)
    RSI_Top =TopDivergence(ta_df["RSI"], raw_df)
    WILLR_Top =TopDivergence(ta_df["WILLR"], raw_df)
    bias_Bottom =BottomDivergence(ta_df["bias"], raw_df)
    k_9_Bottom =BottomDivergence(ta_df["k_9"], raw_df)
    RSI_Bottom =BottomDivergence(ta_df["RSI"], raw_df)
    WILLR_Bottom =BottomDivergence(ta_df["WILLR"], raw_df)
    raw_df.index = pd.to_datetime(raw_df.index)
    Divergences = {'bias_Top':bias_Top, 'k_9_Top':k_9_Top, 'RSI_Top':RSI_Top, 'WILLR_Top':WILLR_Top, 'bias_Bottom':bias_Bottom, 'k_9_Bottom':k_9_Bottom, 'RSI_Bottom':RSI_Bottom, 'WILLR_Bottom':WILLR_Bottom}
    return Divergences

def make_plot(raw_df, ta_df, Divergences):
    tcdf = ta_df[['bb_low','bb_high']]  # DataFrame with two columns
    apd  = mpf.make_addplot(tcdf)
    ap1 = [mpf.make_addplot(ta_df[["bias"]],color='b', ylabel="bias", panel =2)]
    ap2 = [mpf.make_addplot(ta_df[["k_9"]],color='b', ylabel="k_9",  panel =3)]
    ap3 = [mpf.make_addplot(ta_df[["RSI"]],color='b', ylabel="RSI", panel =4)]
    ap4 = [mpf.make_addplot(ta_df[["WILLR"]],color='b', ylabel="WILLR", panel =5)]
    plot_list = []
    if Divergences['bias_Bottom']['value']:
        plot_list.append(mpf.make_addplot(Divergences['bias_Bottom']['value'],type='scatter',markersize=50,color='r',marker='^', panel =2, secondary_y=False))
    if Divergences['k_9_Bottom']['value']:
        plot_list.append(mpf.make_addplot(Divergences['k_9_Bottom']['value'],type='scatter',markersize=50,color='r',marker='^', panel =3, secondary_y=False))
    if Divergences['RSI_Bottom']['value']:
        plot_list.append(mpf.make_addplot(Divergences['RSI_Bottom']['value'],type='scatter',markersize=50,color='r',marker='^', panel =4, secondary_y=False))
    if Divergences['WILLR_Bottom']['value']:
        plot_list.append(mpf.make_addplot(Divergences['WILLR_Bottom']['value'],type='scatter',markersize=50,color='r',marker='^', panel =5, secondary_y=False))
    if Divergences['bias_Top']['value']:
        plot_list.append(mpf.make_addplot(Divergences['bias_Top']['value'],type='scatter',markersize=50,color='g',marker='v', panel =2, secondary_y=False))
    if Divergences['k_9_Top']['value']:
        plot_list.append(mpf.make_addplot(Divergences['k_9_Top']['value'],type='scatter',markersize=50,color='g',marker='v', panel =3, secondary_y=False))
    if Divergences['RSI_Top']['value']:
        plot_list.append(mpf.make_addplot(Divergences['RSI_Top']['value'],type='scatter',markersize=50,color='g',marker='v', panel =4, secondary_y=False))
    if Divergences['WILLR_Top']['value']:
        plot_list.append(mpf.make_addplot(Divergences['WILLR_Top']['value'],type='scatter',markersize=50,color='g',marker='v', panel =5, secondary_y=False))
    fig, axlist = mpf.plot(raw_df,style='yahoo',type='candle',addplot=[apd]+ap1+ap2+ap3+ap4+plot_list, figsize=(11,7), volume=True, returnfig=True)
    fig.tight_layout()
    axlist[2].set_yticklabels([])
    axlist[2].set_ylabel('Volumn')
    axlist[4].set_yticklabels([])
    axlist[6].set_yticklabels([])
    axlist[8].set_yticklabels([])
    axlist[10].set_yticklabels([])
    fig.savefig('stock_ta/static/ta.png',bbox_inches='tight')
    plt.close()

def get_plot(raw_df, willr, bias, rsi):
    ta_df = get_ta_df(raw_df, willr, bias, rsi)
    Divergences =  get_Divergences(ta_df, raw_df)
    make_plot(raw_df, ta_df, Divergences)
    return Divergences