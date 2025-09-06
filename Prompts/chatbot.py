from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, load_prompt
import os
from langchain_core.messages import  AIMessage, HumanMessage, SystemMessage
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
model= GoogleGenerativeAI(model="gemini-2.5-pro")
# chat_history = []

# while True:
#     user_input = input("Enter your query (or type 'exit' to quit): ")
#     chat_history.append(user_input)
#     if user_input.lower() == 'exit':
#         break
#     response = model.invoke(chat_history)
#     chat_history.append(response)
#     print("Response:", response)

chat_history = [ SystemMessage(content="You are a helpful assistant ")]
while True:
    user_input = input("Enter your query (or type 'exit' to quit): ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response))
    print("Response:", response)