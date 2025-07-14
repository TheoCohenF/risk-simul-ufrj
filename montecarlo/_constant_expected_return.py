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
        interval = last_date - self._historical_prices.index[-2]
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

    def plot_ytd_return(self):
        """Plota retorno realizado e simulado para o ano corrente."""
        # Datas do início e fim do ano atual
        from datetime import datetime

        hist_prices = self._historical_prices
        current_year = pd.Timestamp.today().year
        ytd_start = pd.Timestamp(f"{current_year}-01-01")
        ytd_end = hist_prices.index[-1]

        # Filtro: preços de fechamento desde 1º janeiro até o último dado
        hist_ytd = hist_prices[hist_prices.index >= ytd_start]

        if len(hist_ytd) < 2:
            raise ValueError("Não há dados suficientes para o ano corrente.")

        # Retorno realizado YTD (de 1º jan até último fechamento disponível)
        realized_return = hist_ytd.iloc[-1] / hist_ytd.iloc[0] - 1

        # Retorno simulado YTD: compara o último preço histórico com o último simulado
        if self._future_prices is None:
            raise ValueError("Execute run() antes.")

        # Assume que o último preço simulado corresponde ao final do horizonte futuro
        simulated_return = self._future_prices.iloc[-1] / hist_prices.iloc[-1] - 1

        # Gráfico comparativo
        labels = ["Realizado (YTD)", "Simulado (Próx. 21 dias)"]
        returns = [realized_return, simulated_return]
        colors = ["tab:blue", "tab:orange"]

        plt.figure(figsize=(7, 5))
        bars = plt.bar(labels, [100*r for r in returns], color=colors, alpha=0.8)
        plt.ylabel("Retorno (%)")
        plt.title(
            f"Retorno Realizado (YTD) vs Simulado ({len(self._future_prices)} dias)\n{self._ticker.ticker}"
        )

        # Adiciona valores em cima das barras
        for bar, val in zip(bars, returns):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val*100:.2f}%", 
                    ha="center", va="bottom", fontsize=12)

        plt.ylim(min(0, 100*min(returns))-5, max(100*max(returns), 10)+5)
        plt.show()

    def plotly_correlation_heatmap(corr_matrix, title="Matriz de Correlação"):
        """
        Plota a matriz de correlação usando plotly.
        corr_matrix: DataFrame do pandas (preferencial).
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                colorbar=dict(title='Correlação'),
                text=np.round(corr_matrix.values, 2),
                hovertemplate='(%{x}, %{y}): %{z:.2f}<extra></extra>'
            )
        )

        fig.update_layout(
            title=title,
            xaxis=dict(tickangle=45, side='top'),
            yaxis=dict(autorange='reversed'),  # Mantém diagonal principal de cima para baixo
            width=600,
            height=500
        )
        # Adiciona os valores no centro dos quadrados
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                fig.add_annotation(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.index[i],
                    text=f"{corr_matrix.values[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if abs(corr_matrix.values[i, j]) < 0.5 else "white"),
                    xanchor="center",
                    yanchor="middle"
                )
        fig.show()