import numpy as np
"""
    Fractional difference module:
    made for representing time series data in a compact
    stationary format that still
    represents knowledge of past.
"""


def calc_weights(d,lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w=[1]
    for k in range(1,lags):
        w.append(-w[-1]*((d-k+1))/k)
    w=np.array(w).reshape(-1)
    return w

def cutoff_find(order,cutoff,start_lags): #order is our dearest d, cutoff is 1e-5 for us, and start lags is an initial amount of lags in which the loop will start, this can be set to high values in order to speed up the algo
    val=np.inf
    lags=start_lags
    while abs(val)>cutoff:
        w=calc_weights(order, lags)
        val=w[len(w)-1]
        lags+=1
    return lags 


def differencing(time_series, order=0.33, tau=1e-4):
    """
        Time series differencing using tau.
        Just use 0.33 by default, EDA made it.
    """
    
    # return the time series resulting from (fractional) differencing
    lag_cutoff=(cutoff_find(order,tau,1)) #finding lag cutoff with tau
    
    if lag_cutoff >= len(time_series):
        for e_order in [0.5, 0.7, 0.9]:
            new_lag_cutoff=(cutoff_find(e_order,tau,1))
            
            if new_lag_cutoff < len(time_series):
                lag_cutoff = new_lag_cutoff
                old_order = order
                order = e_order
                break
        if lag_cutoff >= len(time_series):
            raise Exception(f"The series is too small, especially for order: {order}")
        print(f"Warning: increased order from {old_order} to {order}")
    
    weights=calc_weights(order, lag_cutoff)
    res=0
    for k in range(lag_cutoff):
        res += weights[k]*time_series.shift(k)
    return res.iloc[lag_cutoff:]