"""
LangChain core concepts - LECL and Runnables
"""


from dotenv import load_dotenv
load_dotenv()

from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def demo_basic_chain(question: str) -> str:
    """
        Demonstrates a basic chain using LCEL and Runnables

        Args:
            question (str): The question to ask the assistant
        
        Returns:
            str: The assistant's response
    """


    prompt = ChatPromptTemplate.from_template(template="You are snarky assistant. Answer in one sentence. {question}") 
    
    llm = ChatOpenRouter(model='openai/gpt-oss-120b')
    
    parser = StrOutputParser()
    
    #Compose with pipe operator
    #chain = prompt.pipe(llm).pipe(parser)
    chain = prompt | llm | parser

    #Excute chain with input
    result = chain.invoke({'question': question})

    return result

def demo_batch_execuation(questions: list[dict]):
    """
        Demonstrates batch execution of a chain using LCEL and Runnables

        Args:
            questions (list[dict]): A list of questions to ask the assistant
        
        Returns:
            None
    """
    prompt = ChatPromptTemplate.from_template(template="Translate to Italian : {text}")
    llm = ChatOpenRouter(model='openai/gpt-oss-120b')
    parser = StrOutputParser()

    chain = prompt | llm | parser

    result = chain.batch(questions)

    for res in zip(result, questions):
        print(f"Input: {res[1]['text']} --> Output: {res[0]}")

def demo_streaming_execution(question: str):
    """
        Demonstrates streaming execution of a chain using LCEL and Runnables

        Args:
            question (str): The question to ask the assistant
        
        Returns:
            None
    """
    prompt = ChatPromptTemplate.from_template(template="You are snarky assistant. Answer in one sentence. {question}") 
    llm = ChatOpenRouter(model='openai/gpt-oss-120b', streaming=True)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    # Streaming execution
    for chunk in chain.stream({'question': question}):
        print(chunk, end='', flush=True)


demo_streaming_execution("What is the airspeed of a laden swallow?")
