import pandas as pd
import numpy as np
import yfinance as yf

# Value at Risk (VaR)
def calc_portfolio_var(ticker_values, confidence_level=0.95, horizon_days=1, period="5y"):
    """
    Calcula o VaR histórico do portfólio ponderado pelos valores investidos.

    Parâmetros:
        ticker_values (dict): {ticker: valor_investido}
        confidence_level (float): Nível de confiança (ex: 0.95)
        horizon_days (int): Horizonte de risco em dias
        period (str): Período dos preços (default '5y')

    Retorna:
        float: VaR do portfólio (valor positivo, perda potencial)
    """
    price_df = pd.DataFrame()
    for ticker in ticker_values:
        price_df[ticker] = yf.Ticker(ticker).history(period=period, interval="1d")["Close"]
    price_df = price_df.dropna()
    
    returns = price_df.pct_change().dropna()
    
    total_investido = sum(ticker_values.values())
    weights = np.array([valor / total_investido for valor in ticker_values.values()])
    
    # Retorno diário do portfólio (soma ponderada)
    portfolio_returns = returns @ weights

    # Retorno para o horizonte
    horizon_returns = portfolio_returns * np.sqrt(horizon_days)
    var_percentile = 100 * (1 - confidence_level)
    var_value = horizon_returns.quantile(var_percentile / 100)

    # VaR absoluto (quanto em R$ posso perder, não apenas percentual)
    var_abs = var_value * total_investido

    return float(-var_abs)


# Drawdown limit
def calc_portfolio_drawdown(ticker_values, period="5y"):
    """
    Calcula o máximo drawdown do portfólio, ponderado pelos valores investidos.

    Parâmetros:
        ticker_values (dict): {ticker: valor_investido}
        period (str): período dos preços (default '5y')

    Retorna:
        dict: {'max_drawdown': valor, 'start_date': data, 'end_date': data}
    """
    price_df = pd.DataFrame()
    for ticker in ticker_values:
        price_df[ticker] = yf.Ticker(ticker).history(period=period, interval="1d")["Close"]
    price_df = price_df.dropna()

    # Normaliza a evolução de cada ativo desde o início
    normed = price_df / price_df.iloc[0]
    weights = np.array([valor for valor in ticker_values.values()])
    
    # Valor diário do portfólio ao longo do tempo (ponderação explícita)
    portfolio_value = normed.mul(weights, axis=1).sum(axis=1)
    
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    end_date = drawdown.idxmin()
    start_date = portfolio_value.loc[:end_date].idxmax()

    return {
        'max_drawdown': max_drawdown,
        'start_date': start_date,
        'end_date': end_date
    }

# Volatilidade
def calc_volatility(price_series, annualized=True, periods_per_year=252):
    """
    Calcula a volatilidade da série de preços com opção de anualização.

    Parameters:
        price_series (pd.Series): Série de preços históricos.
        annualized (bool): Se True, retorna volatilidade anualizada.
        periods_per_year (int): Número de períodos por ano (default = 252 dias úteis).

    Returns:
        float: Volatilidade como decimal (ex: 0.23 = 23%)
    """
    returns = price_series.pct_change().dropna()
    volatility = returns.std()
    if annualized:
        volatility *= (periods_per_year ** 0.5)
    return volatility
