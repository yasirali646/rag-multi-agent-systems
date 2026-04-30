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


if __name__ == "__main__":    
    parallel_chain_demo()