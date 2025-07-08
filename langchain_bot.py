from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def portofolio_generator_chain(user_question):


    llm = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free",
    openai_api_base="https://openrouter.ai/api/v1",  
    openai_api_key="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"
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