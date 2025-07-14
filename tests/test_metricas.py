import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import metricas as mtc

def test_stress_indicator():
    print("##### Testing Stress Indicator #####")
    sp500 = yf.Ticker("SPY")
    sp500_history = sp500.history(period="5y", interval="1d")["Close"]
    aapl = yf.Ticker("AAPL")
    aapl_history = aapl.history(period="5y", interval="1d")["Close"]
    stress = mtc.calc_stress_indicator(aapl_history, sp500_history)
    print(stress)
    return

def test_sharpe_ratio():
    print("##### Testing Sharpe Ratio #####")
    aapl = yf.Ticker("AAPL")
    aapl_history = aapl.history(period="5y", interval="1d")["Close"]
    sharpe = mtc.calc_sharpe_ratio(aapl_history)
    print(f"Sharpe Ratio (AAPL): {sharpe}")
    return

def test_correlation_matrix():
    print("##### Testing Correlation Matrix #####")
    googl = yf.Ticker("GOOGL")
    googl_history = googl.history(period="5y", interval="1d")["Close"]
    msft = yf.Ticker("MSFT")
    msft_history = msft.history(period="5y", interval="1d")["Close"]
    amzn = yf.Ticker("AMZN")
    amzn_history = amzn.history(period="5y", interval="1d")["Close"]
    correlation_matrix = mtc.calc_correlation_matrix({
        "GOOGL": googl_history,
        "MSFT": msft_history,
        "AMZN": amzn_history,
    })
    print("Correlation Matrix:\n", correlation_matrix)
    return

def test():
    test_stress_indicator()
    test_sharpe_ratio()
    test_correlation_matrix()
    return

if __name__ == "__main__":
    test()
