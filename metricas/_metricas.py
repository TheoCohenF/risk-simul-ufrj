import pandas as pd
import yfinance as yf
import numpy as np

def calc_portfolio_stress_indicator(tickers_dict, ticker_index, lookback=14):
    """
    Calcula o stress indicator de um portfólio de ações em relação a um índice,
    ponderado pelos valores investidos em cada ativo.
    
    Parâmetros:
        tickers_dict (dict): {ticker: valor_investido, ...}
        ticker_index (str): Ticker do índice de referência
        lookback (int): Período de cálculo do estocástico

    Retorna:
        pd.Series: Stress indicador ponderado do portfólio
    """
    # Calcula Stochastic Oscillator do índice
    index_close = yf.Ticker(ticker_index).history(period="5y", interval="1d")["Close"]
    index_low = index_close.rolling(window=lookback).min()
    index_high = index_close.rolling(window=lookback).max()
    index_stoch = ((index_close - index_low) / (index_high - index_low)) * 100

    stress_df = pd.DataFrame()

    for ticker, valor in tickers_dict.items():
        stock_close = yf.Ticker(ticker).history(period="5y", interval="1d")["Close"]
        # Alinha datas
        stock_close = stock_close.reindex(index_close.index).fillna(method='ffill')
        stock_low = stock_close.rolling(window=lookback).min()
        stock_high = stock_close.rolling(window=lookback).max()
        stock_stoch = ((stock_close - stock_low) / (stock_high - stock_low)) * 100

        # Diferença do estocástico
        diff_stoch = stock_stoch - index_stoch
        diff_stoch_low = diff_stoch.rolling(window=lookback).min()
        diff_stoch_high = diff_stoch.rolling(window=lookback).max()
        stress_indicator = ((diff_stoch - diff_stoch_low) / (diff_stoch_high - diff_stoch_low)) * 100

        # Pondera pelo valor investido
        stress_df[ticker] = stress_indicator * valor

    # Soma ponderada pelo total investido
    total_investido = sum(tickers_dict.values())
    portfolio_stress = stress_df.sum(axis=1) / total_investido

    return portfolio_stress

def calc_portfolio_sharpe(ticker_values, risk_free_rate=0.025):
    """
    Calcula o Sharpe Ratio anualizado do portfólio ponderado pelos valores investidos.
    
    Parâmetros:
        ticker_values (dict): {ticker: valor_investido}
        risk_free_rate (float): taxa livre de risco anual (padrão 2,5%)
        
    Retorna:
        float: Sharpe Ratio anualizado do portfólio
    """
    price_df = pd.DataFrame()
    for ticker in ticker_values:
        price_df[ticker] = yf.Ticker(ticker).history(period="5y", interval="1d")["Close"]
    price_df = price_df.dropna()
    
    returns = price_df.pct_change().dropna()
    
    total_investido = sum(ticker_values.values())
    weights = np.array([valor / total_investido for valor in ticker_values.values()])
    
    # Retorno diário do portfólio
    portfolio_returns = returns @ weights

    mean_daily_return = portfolio_returns.mean()
    std_daily_return = portfolio_returns.std()
    risk_free_daily = (1 + risk_free_rate)**(1 / 252) - 1

    sharpe_ratio = ((mean_daily_return - risk_free_daily) / std_daily_return) * np.sqrt(252)
    return sharpe_ratio


def calc_correlation_matrix(ticker_values, period="5y"):
    """
    Calcula a matriz de correlação entre os retornos de tickers fornecidos.

    Parâmetros:
        ticker_values (dict): {ticker: valor_investido}
        period (str): Período para download dos preços (padrão "5y")

    Retorna:
        pd.DataFrame: Matriz de correlação dos retornos dos ativos
    """
    price_df = pd.DataFrame()
    for ticker in ticker_values:
        price_df[ticker] = yf.Ticker(ticker).history(period=period, interval="1d")["Close"]
    price_df = price_df.dropna()
    returns_df = price_df.pct_change().dropna()
    return returns_df.corr()
