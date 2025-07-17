import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_bot import portofolio_generator_chain, json_generator_chain
from metricas.Metricas2 import calc_portfolio_var, calc_portfolio_drawdown, calc_volatility  
from plots.plots_functions import plotly_correlation_heatmap, plotly_multi_ytd_historical_vs_simulation, plot_volatility_vs_expected_return, plotly_portfolio_simulation, plot_return_distribution
import numpy as np

#rom simulation import simulation_main

load_dotenv()

st.set_page_config(layout="wide", page_title="Simulador de Portfólio", page_icon="📈")



# === Header com logo ===
def header():
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            "<h1 style='margin-top: 170px;'>Simulador de Portfólio e Risco</h1>",
            unsafe_allow_html=True
        )

    with col2:
        st.image("/Users/theocohen/Desktop/theo/risk-simul-ufrj/ufrj-vertical-cor-rgb-telas.png", width=270)

def chat_box_button():
    st.markdown("""
        <style>
        div.stButton > button {
            font-size: 20px !important;
            font-weight: 600 !important;
            padding: 20px 40px !important;
            border-radius: 12px !important;
            background-color:  !important;
            color: black !important;
            width: 200px !important;
            height: 90px !important;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #F0F0F0  !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("")  # espaço se quiser adicionar algum texto acima do botão
        run = st.button('🚀 Gerar Simulação', use_container_width=True)
    
    return run

def edit_button():
    st.markdown("""
        <style>
        div.stButton > button {
            font-size: 20px !important;
            font-weight: 600 !important;
            padding: 20px 40px !important;
            border-radius: 12px !important;
            background-color:  !important;
            color: black !important;
            width: 200px !important;
            height: 90px !important;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #F0F0F0  !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("")
        run = st.button('✏️ Editar Portfólio', use_container_width=True)

    return run


def chat_box():
    st.markdown("""
        <style>
        .custom-input input {
            font-size: 20px !important;
            font-weight: 600 !important;
            padding: 15px 20px !important;
            border: 2px solid #4A90E2 !important;
            border-radius: 10px !important;
            background-color: #F7F9FC !important;
            width: 300px !important;
            height: 150px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🎯 Perfil do Investidor e Objetivo Financeiro", unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_input = st.text_area(
            label="",
            placeholder="Descreva seu objetivo financeiro e perfil de risco (ex: conservador, moderado, arrojado)",
            key="custom_input",
        )
        st.markdown('<div class="custom-input"></div>', unsafe_allow_html=True)

    return user_input


def simulation_main():
    
    if st.button("🔙 Voltar para Início"):
        st.session_state.page = "home"
        st.rerun()

    col1, col2, col3 = st.columns([2,1,2])
    with col1:

        st.title("📊 Simulação de Portfólio")

    with col3:
        st.markdown("### ✏️ Deseja ajustar o portfólio?")
        nova_pergunta = st.text_input("Digite uma nova pergunta ou ajuste:", key="ajuste_portfolio")
        col1, col2, col3 = st.columns([1.5, 1, 1.5])
        
        if nova_pergunta.strip():
            texto_editado = portofolio_generator_chain(
                nova_pergunta + "\nConsidere o portfólio sugerido anteriormente:\n" + st.session_state.text
            )
            st.session_state.text = texto_editado
            st.session_state.changes = True
        else:
            st.warning("Digite uma pergunta ou ajuste para editar o portfólio.")

    if "text" not in st.session_state or st.session_state.text.strip() == "":
        st.warning("⚠️ Nenhuma recomendação encontrada. Volte à página inicial e descreva seu perfil.")
        if st.button("🔙 Voltar para Início"):
            st.session_state.page = "home"
            st.switch_page("app.py")
        return

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    
    if st.session_state.get("changes", False):
        ticker_values = json_generator_chain(st.session_state.text, st.session_state.user_input)
        st.session_state.ticker_values = ticker_values
        st.session_state.changes = False 
    elif "ticker_values" not in st.session_state:
   
        st.session_state.ticker_values = json_generator_chain(st.session_state.text, st.session_state.user_input)


    ticker_values = st.session_state.ticker_values

    # ==== Mostrar tickers com nomes bonitos de forma compacta e centralizada ====
    nice_names = {}
    for ticker in ticker_values:
        try:
            info = yf.Ticker(ticker).info
            name = info.get("longName", ticker)
        except:
            name = ticker
        nice_names[name] = ticker_values[ticker]

    # Criar tabela compacta manualmente
    st.subheader("💰 Alocação Sugerida:")
    with st.container():
        
        cols = st.columns(len(nice_names))
        for col, (ativo, valor) in zip(cols, nice_names.items()):
            col.markdown(f"**{ativo}**")
            col.markdown(f"${valor:,.0f}")

    # === Cálculo de métricas ===
    with st.spinner("Calculando métricas financeiras..."):
        try:
            var = calc_portfolio_var(ticker_values, confidence_level=0.95, horizon_days=1, period="10y")
            drawdown = calc_portfolio_drawdown(ticker_values, period="10y")

            
            price_df = pd.DataFrame()
            for ticker in ticker_values:
                price_df[ticker] = yf.Ticker(ticker).history(period="10y", interval="1d")["Close"]
            price_df = price_df.dropna()
            normed = price_df / price_df.iloc[0]
            weights = np.array(list(ticker_values.values()))
            port_value = normed.mul(weights, axis=1).sum(axis=1)

            port_returns = port_value.pct_change().dropna()
            volatility = calc_volatility(port_value)
            sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252)
            estresse = drawdown['max_drawdown']

            # Drawdown Limit
            drawdown_limit = -0.40
            within_limit = estresse >= drawdown_limit

            # Início do período
            start_period = price_df.index[0].date()

        

            
            st.markdown("### 📈 Métricas de Risco e Retorno")
            st.divider()
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("📉 VaR", f"${var:,.2f}")
            col2.metric("📊 Volatilidade Anual", f"{volatility * 100:.2f}%")
            col3.metric("📈 Sharpe Ratio", f"{sharpe:.2f}")
            col4.metric("📉 Máx. Drawdown", f"{estresse * 100:.2f}%")
            col5.metric("🕒 Início do Período", f"{start_period}")
            
            st.divider()

            
            row1_col1, row1_col2 = st.columns([1.2, 1.8])
            with row1_col1:
                st.subheader("🔗 Matriz de Correlação")
                try:
                    corr_matrix = price_df.pct_change().dropna().corr()
                    fig_corr = plotly_correlation_heatmap(corr_matrix, title="Correlação dos Ativos")
                    st.markdown("")
                    st.markdown("")
                    st.markdown("")
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao gerar matriz de correlação: {e}")

            with row1_col2:
                st.subheader("📅 Simulação Monte Carlo")
                try:
                    fig, retorno_esperado = plotly_portfolio_simulation(ticker_values, num_simulations=1000, num_days=100, period="5y")
                    st.plotly_chart(fig, use_container_width=True)
                    col6.metric("📈 Ret. Simulação 100 dias", f"{retorno_esperado:.2f}%")


                except Exception as e:
                    st.error(f"Erro ao gerar gráfico de simulações: {e}")

            st.markdown("---")

            row2_col1, row2_col2 = st.columns(2)

            with row2_col1:
                st.subheader("⚖️ Volatilidade vs Retorno Esperado")
                try:
                    tickers = list(ticker_values.keys())
                    fig_vol_ret = plot_volatility_vs_expected_return(
                        ticker_values,
                        period="5y",
                        num_simulations=3000,
                        num_days=252
                    )
                    st.plotly_chart(fig_vol_ret, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico de volatilidade x retorno: {e}")

            with row2_col2:
                st.subheader("📊 Distribuição dos Retornos Simulados")
                try:
                    fig_return_dist = plot_return_distribution(
                        ticker_values,
                        period="5y",
                        num_simulations=3000,
                        num_days=252
                    )
                    st.plotly_chart(fig_return_dist, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao gerar distribuição dos retornos: {e}")
        except Exception as e:
            st.error(f"Erro ao gerar gráfico de simulações: {e}")


# === Resposta do LLM ===
def llm_answer(user_question):
    if user_question.strip() == "":
        return ""
    return portofolio_generator_chain(user_question)

# === Função principal ===
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        header()
        st.markdown("---")
        user_question = chat_box()
        st.session_state.user_question = user_question

        if user_question:
            texto_llm = llm_answer(user_question)
            st.session_state.text = texto_llm
            if texto_llm:
                st.markdown("### 💡 Recomendações de Portfólio com base no seu perfil:")
                st.markdown("")
                st.markdown(texto_llm)

                st.divider()

                st.markdown("A partir dessas recomendações, podemos gerar um simulador de portfólio completo com as sugestões desejadas.")
                col1, col2, col3 = st.columns([1.5, 1, 1.5])
                with col2:
                    st.markdown("")
                    if chat_box_button():
                        st.session_state.page = 'simulacao'
                        st.rerun()  

    elif st.session_state.page == 'simulacao':
        simulation_main()



if __name__ == '__main__':
    main()
