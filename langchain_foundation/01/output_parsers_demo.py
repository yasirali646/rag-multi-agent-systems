from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

from pydantic import BaseModel, Field

load_dotenv()

def str_out_parser_demo():
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_template("write a short poem about {topic}")

    llm = ChatOpenRouter(model="openai/gpt-oss-120b", temperature=0)

    chain = prompt | llm | parser

    response = chain.invoke({"topic" : "nature"})

    print(response)



# JsonOutputParser Example
def json_out_parser_demo():
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template("Return a JSON object with 'name' and 'age' for: {description}")

    llm = ChatOpenRouter(model="openai/gpt-oss-120b", temperature=0)

    chain = prompt | llm | parser

    response = chain.invoke({"description" : "A 25-year-old developer named Yasir"})

    print(response)
    print(type(response))



# PydanticOutputParser Example
def pydantic_output_parser_demo():

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")
        occupation: str = Field(description="The person's occupation")


    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name', 'age' and 'occupation' for: {description}"
        ).partial(format_instruction=parser.get_format_instructions())
    
    llm = ChatOpenRouter(model="openai/gpt-oss-120b", temperature=0)

    chain = prompt | llm | parser

    result = chain.invoke({"description" : "A 30-year-old artist Fatima"})

    print(result)
    print(type(result))


# Structured Output
def structured_output():
    class MovieReview(BaseModel):
        title : str = Field(description="The title of the movie")
        review: str = Field(description="A brief review of the movie")
        rating: int = Field(description="The rating of the movie")


    llm = ChatOpenRouter(model="openai/gpt-oss-120b", temperature=0)
    
    # Bind the schema to the model
    structured_model = llm.with_structured_output(MovieReview)

    result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")
    print(result)
    print(type(result))


structured_output()