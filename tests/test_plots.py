import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import plots
import plotly.io as pio
pio.renderers.default = "browser"
import metricas as mtc
import yfinance as yf

def test_plotly_correlation_heatmap():
    print("##### Testing Plotly Correlation Heatmap #####")
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

    plots.plotly_correlation_heatmap(correlation_matrix)
    return

def test_plotly_multi_ytd_historical_vs_simulation():
    print("##### Testing Plotly Multi YTD Historical vs Simulation #####")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    ticker_hist_dict = {}
    
    for ticker in tickers:
        hist_prices = yf.Ticker(ticker).history(period="5y", interval="1d")["Close"]
        ticker_hist_dict[ticker] = hist_prices

    plots.plotly_multi_ytd_historical_vs_simulation(ticker_hist_dict)
    return

def test_plot_volatility_vs_expected_return():
    print("##### Testing Volatility vs Expected Return #####")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    plots.plot_volatility_vs_expected_return(tickers)

    return

def test():
    test_plot_volatility_vs_expected_return()
    return

if __name__ == "__main__":
    test()
