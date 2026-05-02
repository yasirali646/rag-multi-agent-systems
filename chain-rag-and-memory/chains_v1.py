from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel, 
    RunnablePassthrough, 
    RunnableLambda,
    RunnableBranch
)


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
    """ A chain that demonstrates the use of RunnablePassthrough to pass the same input to multiple branches of a parallel chain. 
    
        *Big Picture*
        
        Question
            ↓
        Retrieve Context
            ↓
        Combine:
            - original question
            - retrieved docs
            ↓
        Send to LLM
    
    
    """

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

def branching_chain_demo():
    """ A chain that demonstrates the use of branching to route inputs to different branches based on a condition. 
    
        *Big Picture*

        User Question
            ↓
        Classifier LLM
            ↓
        Is it coding-related?
            / \
        Yes  No
        /      \
        Code     General
        Prompt    Prompt    
    
    """

    code_prompt = ChatPromptTemplate.from_template(
        template="""You are a coding expert. Help with: {input}"""
    )

    general_prompt = ChatPromptTemplate.from_template(
        template="""You are a helpful assistant. answer: {input}"""
    )

    classify_prompt = ChatPromptTemplate.from_template(
        template="""Classify this as 'code' or 'general': {input}\nReturn only the classification."""
    )

    classifier = classify_prompt | model | StrOutputParser()

    # A simple condition function that checks if the classification contains the word "code". In a real application, this could be more complex and involve multiple conditions.
    def is_code_problem(input_dict):
        classification = classifier.invoke(input_dict)
        return "code" in classification.lower()

    """
    RunnableBranch(
        (condition1, runnable1),
        (condition2, runnable2),
        default_runnable
    )
    """
    branch = RunnableBranch(
        (is_code_problem, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser()
    )

    questions = [
        "How do I reverse a list in Python?",
        "What is the capital of France?"
    ]

    for question in questions:
        result = branch.invoke({'input': question})
        print(f"Question: {question}\nAnswer: {result}\n")

if __name__ == "__main__":    
    # parallel_chain_demo()
    # passthrough_chain_demo()
    branching_chain_demo()