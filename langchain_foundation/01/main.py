from dotenv import load_dotenv
load_dotenv()

from langchain_openrouter import ChatOpenRouter


llm = ChatOpenRouter(model='openai/gpt-oss-20b')
res = llm.invoke("Hello")

print(res.content)