import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class ConstantExpectedReturn:
    def __init__(self, ticker="SPY", periods="1y"):
        self._ticker = yf.Ticker(ticker)
        self._historical_data = self._ticker.history(
            period=periods,
            interval="1d")
        self._rng = np.random.default_rng()
        self._historical_prices = self._historical_data["Close"]
        self._difflogs = self._get_difflogs(self._historical_prices)
        self._difflogs_mean = self._difflogs.mean()
        self._difflogs_std = self._difflogs.std()
        self._current_price = self._historical_prices.iloc[-1]
        self._future_prices = None
        return
        
    def get_historical_data(self):
        return self._historical_data
    
    def get_future_prices(self):
        if self._future_prices is None:
            raise ValueError(
                "Run the simulation before trying to get future prices."
            )
        return self._future_prices
    
    def _get_difflogs(self, prices):
        return (np.log(prices.shift(1)) - np.log(prices)).dropna()
    
    def _get_future_prices_sample(self, num_days=21):
        log_returns = self._rng.normal(
            self._difflogs_mean, 
            self._difflogs_std,
            num_days
        )
        future_prices = self._current_price * np.exp(np.cumsum(log_returns))
        last_date = self._historical_prices.index[-1]
        # Use minimum interval between consecutive dates
        interval = self._historical_prices.index.to_series().diff().min()
        future_dates = [
            last_date + (i + 1)*interval for i in range(len(future_prices))
        ]
        return pd.Series(future_prices, index=future_dates)

    def run(self, num_simulations=3_000, num_days=21):
        scenarios = []
        for i in range(num_simulations):
            scenarios.append(self._get_future_prices_sample(num_days))
        self._future_prices = pd.DataFrame(scenarios).mean(axis=0)        
        return
    
    def plot(self):
        if self._future_prices is None:
            raise ValueError(
                "Run the simulation before trying to run the plot() method."
            )
        all_prices = pd.concat([self._historical_prices, self._future_prices])
        # plot as one line, but with two colors
        plt.figure(figsize=(12, 7))
        plt.plot(all_prices, color="gray", alpha=0.3)  # background for context
        plt.plot(
            self._future_prices,
            color="black",
            label="Simulated Future Prices"
        )
        plt.plot(
            self._historical_prices, 
            color="darkgray", 
            label="Historical Prices"
        )
        plt.title(
            f"Historical and Simulated Future Prices of {self._ticker.ticker} "
            f"({self._ticker.info.get('shortName')})"
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        # plt.ylim(bottom=0)
        plt.show()
        return
