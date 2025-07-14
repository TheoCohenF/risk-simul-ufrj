import pandas as pd
import numpy as np

# Value at Risk (VaR)
def calc_var(price_series, confidence_level=0.95, horizon_days=1):
    """
    Calcula o Value at Risk (VaR) utilizando o método de simulação histórica.

    Parameters:
        price_series (pd.Series): Série de preços históricos.
        confidence_level (float): Nível de confiança desejado (ex: 0.95 para 95%).
        horizon_days (int): Quantidade de dias para o horizonte de risco.

    Returns:
        float: VaR como um valor positivo indicando a perda potencial.
    """
    if len(price_series) < horizon_days + 1:
        raise ValueError("Série de preços muito curta para o horizonte dado.")

    returns = price_series.pct_change().dropna()
    horizon_returns = returns * (horizon_days ** 0.5)
    var_percentile = 100 * (1 - confidence_level)
    var_value = horizon_returns.quantile(var_percentile / 100)

    return float(-var_value.iloc[0] if isinstance(var_value, pd.Series) else -var_value)


# Drawdown limit
def calc_drawdown_limit(price_series):
    """
    Calcula o máximo drawdown da série de preços — maior perda a partir de um pico.

    Parameters:
        price_series (pd.Series): Série de preços históricos.

    Returns:
        dict: Contém o valor do drawdown, a data de início e a data de fim.
    """
    rolling_max = price_series.cummax()
    drawdown = (price_series - rolling_max) / rolling_max

    max_drawdown = drawdown.min()
    end_date = drawdown.idxmin()
    if isinstance(end_date, pd.Series):
        end_date = end_date.iloc[0]

    start_date = price_series.loc[:end_date].idxmax()

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
