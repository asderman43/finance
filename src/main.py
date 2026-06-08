import yfinance as yf

def get_data(stocks, start, end):
    data = yf.download(stocks, start=start, end=end)
    return data

def get_returns(data):
    stock_data = data["Close"]
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    return mean_returns, cov_matrix

def str_num(num):
    return f"{num:_.2f}"