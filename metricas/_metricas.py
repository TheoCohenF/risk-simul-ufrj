import pandas as pd

def calc_stress_indicator(stock_close, index_close, lookback=14):
    """
        Calculate the stress indicator based on the Stochastic Oscillator
    for a stock compared to an index.
    
    Parameters:
        stock_close (pd.Series): Closing prices of the stock.
        index_close (pd.Series): Closing prices of the index.
        lookback (int): Lookback period for the Stochastic Oscillator.

    Returns:
        pd.Series: Stress indicator values.
    """
    # Calculate Stochastic Oscillator for stock
    stock_low = stock_close.rolling(window=lookback).min()
    stock_high = stock_close.rolling(window=lookback).max()
    stock_stoch = ((stock_close - stock_low) / (stock_high - stock_low)) * 100
    # Calculate Stochastic Oscillator for index
    index_low = index_close.rolling(window=lookback).min()
    index_high = index_close.rolling(window=lookback).max()
    index_stoch = ((index_close - index_low) / (index_high - index_low)) * 100
    # Calculate Stochastic Oscillator for difference between 
    # stock and index stochastics
    diff_stoch = stock_stoch - index_stoch
    diff_stoch_low = diff_stoch.rolling(window=lookback).min()
    diff_stoch_high = diff_stoch.rolling(window=lookback).max()
    stress_indicator = ((diff_stoch - diff_stoch_low) 
                        / (diff_stoch_high - diff_stoch_low)) * 100
    return stress_indicator

def calc_sharpe_ratio(price_series, risk_free_rate=0.025):
    """
        Calculate the Sharpe Ratio for a given price series.
    
    Parameters:
        price_series (pd.Series): Series of stock prices.
        risk_free_rate (float): Annual risk-free rate (default is 2.5%).
        
    Returns:
        float: Sharpe Ratio.
    """
    returns = price_series.pct_change().dropna()
    expected_return = (1 + returns).prod() - 1
    # Convert to daily expected return
    expected_return = (1 + expected_return)**(1 / len(returns)) - 1
    # Convert to annualized expected return
    expected_return = (1 + expected_return)**252 - 1
    # Convert risk-free rate to daily
    risk_free_rate = (1 + risk_free_rate)**(1 / 252) - 1
    # Calculate Sharpe Ratio
    std_return = returns.std()
    sharpe_ratio = (expected_return - risk_free_rate) / std_return
    return sharpe_ratio

def calc_correlation_matrix(stocks_series_dict):
    """
        Calculate the correlation matrix for a dictionary of stock price series.

    Parameters:
        stocks_series_dict (dict): Dictionary where keys are stock names
            and values are pd.Series of stock prices.
    
    Returns:
        pd.DataFrame: Correlation matrix of the stock prices.
    """
    data = pd.DataFrame(stocks_series_dict)
    return data.corr()
