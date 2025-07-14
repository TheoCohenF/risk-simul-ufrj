import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
    
    def get_historical_prices(self):
        return self._historical_prices
    
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

    def plotly_ytd_historical_vs_simulation(self, num_simulations=3000):
        """
        Plota, com Plotly, o preço histórico YTD e o caminho simulado começando do início do ano corrente.
        """

        hist_prices = self._historical_prices
        current_year = pd.Timestamp.today().year

        # Ajusta timezone
        if hist_prices.index.tz is not None:
            ytd_start = pd.Timestamp(f"{current_year}-01-01", tz=hist_prices.index.tz)
        else:
            ytd_start = pd.Timestamp(f"{current_year}-01-01")
        ytd_prices = hist_prices[hist_prices.index >= ytd_start]

        if len(ytd_prices) < 2:
            raise ValueError("Não há dados suficientes para o ano vigente.")

        prev_year_prices = hist_prices[hist_prices.index < ytd_start]
        if not prev_year_prices.empty:
            start_price = prev_year_prices.iloc[-1]
        else:
            start_price = ytd_prices.iloc[0]
        if len(prev_year_prices) < 2:
            raise ValueError("Não há dados históricos suficientes para simular.")

        train_difflogs = np.log(prev_year_prices).diff().dropna()
        mean = train_difflogs.mean()
        std = train_difflogs.std()
        ytd_dates = ytd_prices.index

        n_days = len(ytd_dates) - 1
        simulations = []
        for _ in range(num_simulations):
            sim_returns = self._rng.normal(
                mean,
                std,
                n_days
            )
            sim_prices = [start_price]
            for r in sim_returns:
                sim_prices.append(sim_prices[-1] * np.exp(r))
            simulations.append(sim_prices)

        simulated_mean = np.mean(simulations, axis=0)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ytd_dates,
            y=ytd_prices.values,
            mode="lines+markers",
            name="Preço Real (YTD)",
            line=dict(color="royalblue", width=3)
        ))

        fig.add_trace(go.Scatter(
            x=ytd_dates,
            y=simulated_mean,
            mode="lines",
            name="Preço Simulado (média)",
            line=dict(color="darkorange", dash="dash", width=3)
        ))

        fig.update_layout(
            title=f"Preço Real vs Simulado em {self._ticker.ticker} ({self._ticker.info.get('shortName', '')}) - {current_year}",
            xaxis_title="Data",
            yaxis_title="Preço",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(0,0,0,0)'),
            hovermode="x unified",
            width=900,
            height=500
        )
        fig.show()