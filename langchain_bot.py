from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI



def portofolio_generator_chain(user_question):


    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",  
    google_api_key="AIzaSyDOjxCVX_u1Qbqo47HuPTxyoHzzivdS9UE"  
) 


    prompt = ChatPromptTemplate.from_messages([
            ("system",
            """ 
            Você é um especialista financeiro que cria portfólios personalizados.

            Você só pode utilizar ativos do S&P 500

            Seu objetivo é criar um portfólio diversificado de acordo com o objetivo e perfil do investidor.

            Por exemplo, se o usuário diz que tem um perfil mais agressivo, você deve passar um grupo x de ações.

            Se ele tem um perfil mais conservador, um outro grupo y.

            Ele também pode passar informações como:
            'Preciso de 100 mil reais até o final do ano para comprar minha casa'.
            Leve em conta tais argumentos para análise de sentimento do perfil do investidor e a ação que você irá escolher


            Use separacao de linhas e markdown e EMOJIS para deixar o texto bonito (muito importante)

    
            """
            ),
            ("human",
            """
            Crie um portfólio diversificado com base perfil: {input}
            (lembre que pode usar apenas ações do S&P 500)
            """
              )
        ])


    chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({"input": user_question})
    

    return answer



def json_generator_chain(text, user_question):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",  
        google_api_key="AIzaSyDOjxCVX_u1Qbqo47HuPTxyoHzzivdS9UE"  
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        Você é um assistente que recebe uma descrição textual e deve extrair informações de ações e valores para investir.

        Sua tarefa é gerar um **objeto JSON** com os **tickers válidos de ações do S&P 500** (conforme usados na biblioteca do Yahoo Finance) e os **valores monetários associados a cada um**.

        Regras:
        - Use apenas ações que façam parte do índice S&P 500.
        - Os tickers devem estar no formato aceito pelo Yahoo Finance (ex: AAPL, MSFT, TSLA, AMZN, etc.).
        - O valor de cada ação deve ser um número (representando o valor a ser investido em dólares).
        - O JSON deve estar corretamente formatado.
        - Não inclua explicações ou comentários, apenas o objeto JSON.
        - COLOQUE APENAS O JSON DOS OBJETOS SUGERIDOS PELO TEXTO DE ENTRADA

        Exemplo de saída:
        json
        {{{{
        "AAPL": 5000,
        "TSLA": 3000
        }}}}
            """),
            ("human",
            "Input do user: {user_question}"
            "Texto de entrada: {input}")
        ])

    chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | JsonOutputParser()
    )

    return chain.invoke({"input": text, "user_question": user_question})
