import streamlit as st
import yfinance as yf
import pandas as pd
import plotly 
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_bot import portofolio_generator_chain
load_dotenv()

st.set_page_config(layout="wide")

def select_asset():


    @st.cache_data
    def load_sp500_tickers():
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        return table[0]['Symbol'].tolist()

    tickers = load_sp500_tickers()

    selected_stocks = st.multiselect("Select the stocks", tickers)


    
    return selected_stocks

def run_simulation(selected_stocks):
    historical_df = pd.DataFrame()

    for stock in selected_stocks:
        
        stock_historic = yf.download(stock, start='2008-01-01', end='2025-06-30')

        
        stock_prices = stock_historic[['Close']].reset_index()
        stock_prices.rename(columns={'Close': stock}, inplace=True)  
        
        
        
        if historical_df.empty:
            historical_df = stock_prices
        else:
            historical_df = pd.merge(historical_df, stock_prices, on='Date', how='inner')


    if isinstance(historical_df.columns, pd.MultiIndex):
        historical_df.columns = [' '.join(col).strip() for col in historical_df.columns.values]
        historical_df.columns = [col.split(' ')[0] for col in historical_df.columns]


    return historical_df
    
def header():

    st.markdown('### Risk Simulator Portfolios')

def chat_box():


    input = st.text_input("Explique o seu objetivo financeiro e perfil de investidor", "")

    return input

def llm_answer(user_question):

    generated_portfolio = portofolio_generator_chain(user_question)
    
    st.write(generated_portfolio)

    




def main():

    header()

    user_question = chat_box()

    teste = llm_answer(user_question)

    st.divider()

    st.markdown('A partir dessas recomendações, podemos gerar um simulador de portfólio completo com as recomendações desejadas')
    run = st.button('Run Simulation')
    # selected_stocks = select_asset()


    # st.markdown('')

    # st.divider()
    # if selected_stocks:
    #     cols = st.columns(len(selected_stocks))

    # for i, stock in enumerate(selected_stocks):
    #     with cols[i]:
    #         ticker = yf.Ticker(stock)
    #         st.write(f"### {ticker.info['shortName']}")
            
    # cols = st.columns(3)

    # with cols[1]:
    #     run = st.button('Run Simulation')
    


    # if selected_stocks and run:
    #     df = run_simulation(selected_stocks)

    #     # Plot with Plotly
    #     fig = go.Figure()

    #     for stock in selected_stocks:
    #         print('stock',df[stock])
    #         fig.add_trace(go.Scatter(x=df['Date'], y=df[stock], mode='lines', name=stock))

    #     fig.update_layout(
    #         title='Historical Stock Prices',
    #         xaxis_title='Date',
    #         yaxis_title='Price',
    #         template='plotly_dark',
    #         hovermode='x unified',
    #         width=3000,    
    #         height=600
    #     )

    #     st.plotly_chart(fig, use_container_width=True)





if __name__ =='__main__':
    main()
