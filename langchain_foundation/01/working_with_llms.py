"""
Working with LLMs in LanChain
Multiple providers, configuration, streaming, and cost optimization
"""
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def demo_messages_stream():
    model = ChatOpenRouter(model='openai/gpt-oss-120b', temperature=0.7, max_tokens=100, streaming=True)

    messages = [
        SystemMessage(content="You are a Gen-Z. Always respond in a casual and trendy way."),
        HumanMessage(content="Hello")
    ]

    for chunk in model.stream(messages):
        print(chunk.content, end='', flush=True)


def demo_messages():
    model = ChatOpenRouter(model='openai/gpt-oss-120b', temperature=0.7, max_tokens=100)

    messages = [
        SystemMessage(content="You are a Gen-Z. Always respond in a casual and trendy way."),
        HumanMessage(content="My name is Yasir.")
    ]

    response = model.invoke(messages)

    messages.append(response)

    messages.append(HumanMessage(content="What is my name?"))

    response = model.invoke(messages)

    print(response.content)


demo_messages()