import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from montecarlo import ConstantExpectedReturn
from metricas import calc_volatility

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
        yaxis=dict(autorange='reversed'),
        width=600,
        height=500
    )

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
    return fig

def plotly_multi_ytd_historical_vs_simulation(ticker_value_dict, num_simulations=3000, period="5y"):

    current_year = pd.Timestamp.today().year
    fig = go.Figure()
    ticker_hist_dict = {}

    for ticker in ticker_value_dict:
        df = yf.Ticker(ticker).history(period=period, interval="1d")["Close"]
        if len(df) < 2:
            print(f"[AVISO] Ticker {ticker}: Não há dados suficientes. Pulando.")
            continue
        ticker_hist_dict[ticker] = df
    
    for ticker, hist_prices in ticker_hist_dict.items():

        if hist_prices.index.tz is not None:
            ytd_start = pd.Timestamp(f"{current_year}-01-01", tz=hist_prices.index.tz)
        else:
            ytd_start = pd.Timestamp(f"{current_year}-01-01")
        ytd_prices = hist_prices[hist_prices.index >= ytd_start]

        if len(ytd_prices) < 2:
            print(f"[AVISO] Ticker {ticker}: Não há dados suficientes para o ano vigente. Pulando.")
            continue

        prev_year_prices = hist_prices[hist_prices.index < ytd_start]
        if not prev_year_prices.empty:
            start_price = prev_year_prices.iloc[-1]
        else:
            start_price = ytd_prices.iloc[0]
        if len(prev_year_prices) < 2:
            print(f"[AVISO] Ticker {ticker}: Não há dados históricos suficientes para simular. Pulando.")
            continue

        train_difflogs = np.log(prev_year_prices).diff().dropna()
        mean = train_difflogs.mean()
        std = train_difflogs.std()
        ytd_dates = ytd_prices.index

        n_days = len(ytd_dates) - 1
        simulations = []
        rng = np.random.default_rng()
        for _ in range(num_simulations):
            sim_returns = rng.normal(
                mean,
                std,
                n_days
            )
            sim_prices = [start_price]
            for r in sim_returns:
                sim_prices.append(sim_prices[-1] * np.exp(r))
            simulations.append(sim_prices)

        simulated_mean = np.mean(simulations, axis=0)
        sim_dates = np.insert(ytd_dates.values, 0, ytd_dates[0])


        fig.add_trace(go.Scatter(
            x=ytd_dates,
            y=ytd_prices.values,
            mode="lines",
            name=f"{ticker} real",
            line=dict(width=2),
            showlegend=True
        ))


        fig.add_trace(go.Scatter(
            x=sim_dates,
            y=simulated_mean,
            mode="lines",
            name=f"{ticker} simulado",
            line=dict(width=2, dash="dash"),
            showlegend=True
        ))

    fig.update_layout(
        title=f"Preços Reais vs Simulados (YTD) — {current_year}",
        xaxis_title="Data",
        yaxis_title="Preço",
        legend=dict(x=0, y=1.05, orientation="h"),
        hovermode="x unified",
        width=950,
        height=600
    )
    return fig

def plot_volatility_vs_expected_return(ticker_list, period="2y", num_simulations=3000, num_days=252):
    """
    Plota Volatilidade Anualizada (X) vs Retorno Esperado (Y) usando simulação de Monte Carlo.
    - ticker_list: lista de tickers (ex: ["PETR4.SA", "VALE3.SA"])
    - period: período de histórico para estimar retornos (ex: "2y")
    - num_simulations: número de simulações de Monte Carlo
    - num_days: horizonte da simulação (em dias úteis). Para anual, use 252.
    """
    results = []

    for ticker in ticker_list:
        try:
            data = yf.Ticker(ticker).history(period=period, interval="1d")["Close"]
            data = data.dropna()

            if len(data) < 2:
                print(f"[AVISO] Dados insuficientes para {ticker}")
                continue

           
            daily_returns = data.pct_change().dropna()
            log_returns = np.log(1 + daily_returns)

            mu = log_returns.mean()
            sigma = log_returns.std()
            start_price = data.iloc[-1]


            
            rng = np.random.default_rng()
            all_simulations = np.zeros((num_simulations, num_days + 1))
            all_simulations[:, 0] = start_price

            for i in range(num_simulations):
                sim_returns = rng.normal(mu, sigma, num_days)
                sim_prices = [start_price]
                for r in sim_returns:
                    sim_prices.append(sim_prices[-1] * (1 + r))
                all_simulations[i, :] = sim_prices

           
            final_returns = (all_simulations[:, -1] / all_simulations[:, 0]) - 1
            expected_return = final_returns.mean()  

            
            annual_vol = sigma * np.sqrt(252)

            results.append((ticker, annual_vol, expected_return))

        except Exception as e:
            print(f"[ERRO] {ticker}: {e}")
            continue

    
    fig = go.Figure()
    colors = ['#1f77b4', '#aec7e8', '#d62728', '#ff9896', '#2ca02c', '#98df8a']

    for idx, (ticker, vol, ret) in enumerate(results):
        fig.add_trace(go.Scatter(
            x=[vol * 100],
            y=[ret * 100],
            mode="markers+text",
            marker=dict(size=16, color=colors[idx % len(colors)]),
            text=[ticker],
            name=ticker,
            textposition="top center"
        ))

    fig.update_layout(
        title=f"Volatilidade Anualizada vs Retorno Esperado (Horizonte: {num_days} dias úteis)",
        xaxis_title="Volatilidade Anualizada (%)",
        yaxis_title="Retorno Esperado (%)",
        hovermode="closest",
        width=900,
        height=550
    )

    return fig


def plotly_portfolio_simulation(ticker_value_dict, num_simulations=3000, num_days=45, period="5y"):


 
    price_df = pd.DataFrame()
    for ticker in ticker_value_dict:
        price_df[ticker] = yf.Ticker(ticker).history(period=period, interval="1d")["Close"]
    price_df.dropna(inplace=True)

     
    returns = price_df.pct_change().dropna()

   
    total_value = sum(ticker_value_dict.values())
    weights = np.array([ticker_value_dict[ticker] / total_value for ticker in price_df.columns])

    
    portfolio_returns = returns.dot(weights)

    
    portfolio_price = (1 + portfolio_returns).cumprod()

    
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()

    # 
    last_price = portfolio_price.iloc[-1]

    
    sim_dates = pd.bdate_range(start=portfolio_price.index[-1], periods=num_days + 1)[1:]

    rng = np.random.default_rng()

    
    all_simulations = np.zeros((num_simulations, num_days))

    

    
    for i in range(num_simulations):
        sim_returns = rng.normal(mu, sigma, num_days)
        sim_prices = [last_price]
        for r in sim_returns:
            sim_prices.append(sim_prices[-1] * (1 + r))
        all_simulations[i, :] = sim_prices[1:]

    retornos_percentuais = (all_simulations[:, -1] - last_price) / last_price
    retorno_esperado_pct = np.mean(retornos_percentuais) * 100 

    fig = go.Figure()

  
    fig.add_trace(go.Scatter(
        x=portfolio_price.index,
        y=portfolio_price.values * total_value,
        mode="lines",
        name="Portfólio Real",
        line=dict(width=3)
    ))


    for i in range(num_simulations):
        fig.add_trace(go.Scatter(
            x=sim_dates,
            y=all_simulations[i, :] * total_value,
            mode="lines",
            line=dict(color='rgba(0,0,255,0.03)'),
            showlegend=False,
            hoverinfo='skip', 

        ))

    mean_sim = np.mean(all_simulations, axis=0) * total_value
    fig.add_trace(go.Scatter(
        x=sim_dates,
        y=mean_sim,
        mode="lines",
        name="Média Simulada",
        line=dict(color='red', width=3, dash='dash')
    ))

    fig.update_layout(
        title=f"Simulação Monte Carlo do Portfólio para {num_days} dias",
        xaxis_title="Data",
        yaxis_title="Valor do Portfólio (USD)",
        hovermode="x unified",
        width=950,
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig, retorno_esperado_pct


def plot_return_distribution(ticker_value_dict, num_simulations=3000, num_days=252, period="5y", lower_quantile=0.01, upper_quantile=0.99):

    price_df = pd.DataFrame()
    for ticker in ticker_value_dict:
        price_df[ticker] = yf.Ticker(ticker).history(period=period, interval="1d")["Close"]
    price_df.dropna(inplace=True)

    returns = price_df.pct_change().dropna()

    total_value = sum(ticker_value_dict.values())
    weights = np.array([ticker_value_dict[ticker] / total_value for ticker in price_df.columns])

    portfolio_returns = returns.dot(weights)

    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()

    portfolio_price = (1 + portfolio_returns).cumprod()
    last_price = portfolio_price.iloc[-1]

    rng = np.random.default_rng()
    sim_returns = rng.normal(mu, sigma, size=(num_simulations, num_days))

    sim_prices = np.zeros_like(sim_returns)
    sim_prices[:, 0] = last_price * (1 + sim_returns[:, 0])
    for day in range(1, num_days):
        sim_prices[:, day] = sim_prices[:, day-1] * (1 + sim_returns[:, day])

    final_returns = (sim_prices[:, -1] - last_price) / last_price

    # Filtrando outliers pelos quantis
    lower_bound = np.quantile(final_returns, lower_quantile)
    upper_bound = np.quantile(final_returns, upper_quantile)
    filtered_returns = final_returns[(final_returns >= lower_bound) & (final_returns <= upper_bound)]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=filtered_returns * 100,
        nbinsx=60,
        marker_color='#1f77b4',
        opacity=0.75,
        name='Retornos Simulados (%)'
    ))

    fig.update_layout(
        title=f"Distribuição dos Retornos Simulados do Portfólio ({num_days} dias) - Sem Outliers",
        xaxis_title="Retorno Simulado (%)",
        yaxis_title="Frequência",
        bargap=0.1,
        width=700,
        height=500,
        template='plotly_white'
    )

    return fig
