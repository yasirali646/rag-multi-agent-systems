from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openrouter import ChatOpenRouter


def marketing_tagline(product_name: str, audience: str) -> str:
    """
    Generates a marketing tagline for a given product and target audience using LCEL and Runnables
    """

    prompt = ChatPromptTemplate.from_template(template="""
    You are a marketing expert. Generate a catchy tagline for the following product and target audience. Keep it under 20 words.
    
    Inputs:
        Product Name: {product_name}
        Target Audience: {audience}                                                 
    
    Example:
        Product Name: AI Course
        Target Audience: Developers
    
    "Learn AI with hands-on projects and expert guidance. Enroll now and become an AI pro!"                                                 
    """)
    model = ChatOpenRouter(model='openai/gpt-oss-120b', temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    response = chain.invoke({'product_name': product_name, 'audience' : audience})

    print(response)


marketing_tagline(product_name="Beauty Cream", audience="Girls")
