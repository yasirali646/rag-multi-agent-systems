"""
Project: Smart Q&A Bot
A production-ready question-answering bot with structured output parsing, error handling, and logging. 
"""

import os
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter

from langsmith import traceable, Client

from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

# -- LangSmith Configuration --
if os.getenv("LANGSMITH_API_KEY"):
    print(f"LangSmith configured. - Project: {os.getenv('LANGSMITH_PROJECT')}")



# Schema Defination
class QAResponse(BaseModel):
    answer: str = Field(..., description="The answer to the user's question.")
    confidence: str = Field(None, description="Confidence level: high, medium, low.")
    resoning: str = Field(None, description="The reasoning process behind the answer.")
    follow_up_questions: List[str] = Field(
        description="List of follow-up questions for further clarification.",
        default_factory=list
    )
    sources_needed: bool = Field(
        description="Indicates if the answer requires citing sources.",
        default=False
    )


# Bot Implementation
class SmartQABot:
    """
    A smart Q&A bot that provides concise answers with confidence levels, reasoning, and follow-up questions.
    """
    def __init__(self,
                 model_name: str = "openai/gpt-oss-120b",
                 temperature: float = 0.7
                ):
        self.model = ChatOpenRouter(model=model_name, temperature=temperature).with_structured_output(QAResponse)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a knowledgable Q&A assistant.
                 
                 Your guidlines:
                    - Provide concise and accurate answers.
                    - Be transparent about your confidence level (high, medium, low). - set to low if you are unsure or the question is ambiguous.
                    - Include your reasoning process in the response.
                    - Suggest follow-up questions if the user's question is broad or could be clarified.
                    - Indicate if citing sources is necessary for the answer.
                 
                 Always respond with accurate, helpful information.
                 
                 """),
                ("human", "{question}")
            ]
        )

        self.chain = self.prompt_template | self.model

    @traceable(name="SmartQABot.ask", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            print(f"Error processing the question: {e}")
            return QAResponse(
                answer="Sorry, I encountered an error while processing your question.",
                confidence="low",
                resoning=str(e),
                follow_up_questions=["Could you please try again later?"],
                sources_needed=True
            )
        
    @traceable(name="SmartQABot.ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        """ Ask multiple questions in a batch and return their responses. """
        inputs = [{"question": q} for q in questions]
        return self.chain.batch(inputs)


# Example Usage
def demo_bot_batch():
    bot = SmartQABot()

    questions = [
        "What is the capital of France?",
        "Can you explain the theory of relativity in simple terms?",
        "What are the health benefits of green tea?",
        "How does a blockchain work?",
        "What is the meaning of life?"
    ]

    print("=" * 50)
    print("Smart Q&A Bot Demo")
    print("=" * 50)


    for question in questions:

        print(f"Question: {question}")
        print("-" * 50)

        response = bot.ask(question)

        print("Answer:", response.answer)
        print("Confidence:", response.confidence)
        print("Reasoning:", response.resoning)
        print("Follow-up Questions:", response.follow_up_questions)
        print("Sources Needed:", response.sources_needed)
        print("-" * 50)

def demo_bot():
    bot = SmartQABot()

    question = "What time is it in Tokyo right now?"

    print("=" * 50)
    print("Smart Q&A Bot Demo")
    print("=" * 50)

    print(f"Question: {question}")
    print("-" * 50)

    response = bot.ask(question)

    print("Answer:", response.answer)
    print("Confidence:", response.confidence)
    print("Reasoning:", response.resoning)
    print("Follow-up Questions:", response.follow_up_questions)
    print("Sources Needed:", response.sources_needed)
    print("-" * 50)

if __name__ == "__main__":
    
    try:
        demo_bot()
    finally:
        Client().flush()