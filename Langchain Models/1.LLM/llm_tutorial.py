from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",   
    api_key=api_key
)

# Invoke means to call or execute a function or method.
result = llm.invoke("What is the capital of India?")

print(result.content)
