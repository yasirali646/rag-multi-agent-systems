from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


load_dotenv()

model = ChatOpenRouter(model="openai/gpt-oss-20b", temperature=0)


def parallel_chain_demo():
    summarize_prompt = ChatPromptTemplate.from_template(
        template="""You are a helpful assistant that summarizes the following text: {text}"""
    )

    keywords_prompt = ChatPromptTemplate.from_template(
        template="""You are a helpful assistant that extracts keywords from the following text: {text}"""
    )

    sentiment_prompt = ChatPromptTemplate.from_template(
        template="""You are a helpful assistant that determines the sentiment of the following text: {text}"""
    )

    parser = StrOutputParser()

    chain = RunnableParallel(
        summarize = summarize_prompt | model | parser,
        keyword = keywords_prompt | model | parser,
        sentiment = sentiment_prompt | model | parser
    )

    text = "LangSmith is a great tool for debugging and monitoring your language models! It provides a user-friendly interface and powerful features to help you understand and improve your models."
    results = chain.invoke({'text' : text})

    print(results)


def passthrough_chain_demo():
    """ A chain that demonstrates the use of RunnablePassthrough to pass the same input to multiple branches of a parallel chain. """

    prompt = ChatPromptTemplate.from_template(
        template="""You are a helpful assistant that answers the following question based on the provided 
        context: {context}\n
        Question: {question}"""
    )    
    def fake_retriever(query):
        return "LangChain is developed by LangSmith, a company that provides tools for debugging and monitoring language models."


    chain = (
        RunnableParallel(
            context = RunnableLambda(fake_retriever),
            question = RunnablePassthrough()
        )
        | RunnableLambda(
            lambda inputs: {'context' : inputs['context'], 
                            'question' : inputs['question']['question']} 
        )
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke({'question': "Who developed LangChain?"})
    print(result)


if __name__ == "__main__":    
    # parallel_chain_demo()
    passthrough_chain_demo()