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
    """
    Plota, com Plotly, os preços históricos YTD e os caminhos simulados para múltiplos tickers.
    ticker_hist_dict: dict {ticker: pandas.Series de preços}
    """
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
        # Ajusta timezone
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

        # Linha do preço real
        fig.add_trace(go.Scatter(
            x=ytd_dates,
            y=ytd_prices.values,
            mode="lines",
            name=f"{ticker} real",
            line=dict(width=2),
            showlegend=True
        ))

        # Linha da média simulada
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

def plot_volatility_vs_expected_return(ticker_list, periods="2y", num_simulations=3000, num_days=21):
    """
    Plota volatilidade (X) x retorno esperado futuro (Y) usando simulação de Monte Carlo para cada ativo.
    - ticker_list: lista de tickers (ex: ["PETR4.SA", "VALE3.SA"])
    - periods: período do histórico usado em cada ativo (ex: "2y")
    - num_simulations: número de simulações para estimar retorno futuro
    - num_days: horizonte da simulação de retorno futuro
    """
    results = []
    for ticker in ticker_list:
        try:
            # Instancia a classe e pega preços históricos
            model = ConstantExpectedReturn(ticker, periods)
            hist_prices = model.get_historical_prices()
            # Calcula volatilidade anualizada
            vol = calc_volatility(hist_prices)
            
            model.run(num_simulations, num_days)
            future_prices = model.get_future_prices()
            start_price = hist_prices.iloc[-1]
            expected_return = (future_prices.iloc[-1] / start_price) - 1

            results.append((ticker, vol, expected_return))
        except Exception as e:
            print(f"[AVISO] Problema ao processar {ticker}: {e}")
            continue

    # Plot com Plotly
    fig = go.Figure()
    for ticker, vol, ret in results:
        fig.add_trace(go.Scatter(
            x=[vol * 100],  # em %
            y=[ret * 100],  # em %
            mode="markers+text",
            marker=dict(size=15),
            name=ticker,
            text=[ticker],
            textposition="top center"
        ))

    fig.update_layout(
        title=f"Volatilidade Anualizada vs Retorno Esperado ({num_days} dias)",
        xaxis_title="Volatilidade Anualizada (%)",
        yaxis_title=f"Retorno Esperado ({num_days} dias) (%)",
        hovermode="closest",
        width=850,
        height=550
    )
    return fig

def plotly_portfolio_simulation(ticker_value_dict, num_simulations=3000, num_days=45, period="5y"):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import yfinance as yf

    # Baixa preços históricos
    price_df = pd.DataFrame()
    for ticker in ticker_value_dict:
        price_df[ticker] = yf.Ticker(ticker).history(period=period, interval="1d")["Close"]
    price_df.dropna(inplace=True)

    # Retornos diários
    returns = price_df.pct_change().dropna()

    # Pesos normalizados
    total_value = sum(ticker_value_dict.values())
    weights = np.array([ticker_value_dict[ticker] / total_value for ticker in price_df.columns])

    # Retorno diário do portfólio ponderado
    portfolio_returns = returns.dot(weights)

    # Preço do portfólio normalizado para 1
    portfolio_price = (1 + portfolio_returns).cumprod()

    # Média e desvio padrão dos retornos
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()

    # Último preço real do portfólio
    last_price = portfolio_price.iloc[-1]

    # Datas para simulação (dias úteis após último dado)
    sim_dates = pd.bdate_range(start=portfolio_price.index[-1], periods=num_days + 1)[1:]

    rng = np.random.default_rng()

    # Criar array para armazenar todas as simulações
    all_simulations = np.zeros((num_simulations, num_days))

    # Simular retornos e calcular preços para cada simulação
    for i in range(num_simulations):
        sim_returns = rng.normal(mu, sigma, num_days)
        sim_prices = [last_price]
        for r in sim_returns:
            sim_prices.append(sim_prices[-1] * (1 + r))
        all_simulations[i, :] = sim_prices[1:]  # ignorar o primeiro preço (last_price)

    fig = go.Figure()

    # Linha do portfólio real (com valor investido)
    fig.add_trace(go.Scatter(
        x=portfolio_price.index,
        y=portfolio_price.values * total_value,
        mode="lines",
        name="Portfólio Real",
        line=dict(width=3)
    ))

    # Traçar todas simulações com transparência
    for i in range(num_simulations):
        fig.add_trace(go.Scatter(
            x=sim_dates,
            y=all_simulations[i, :] * total_value,
            mode="lines",
            line=dict(color='rgba(0,0,255,0.03)'),
            showlegend=False,
            hoverinfo='skip',  # <- ESSENCIAL: impede que apareça no hover

        ))

    # Calcular e plotar média simulada
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

    return fig
