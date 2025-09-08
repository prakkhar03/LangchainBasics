# Output parsers help in converting the output of the LLM into a specific format
# Structured output helps in defining a schema for the output of the LLM
# Types of output parsers:
# 1. String output parser
# 2. JSON output parser
# 3. Pydantic output parser
# 4. Structured output parser

# String output parser means the output of the LLM is a string

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
# Make sure you have set GOOGLE_API_KEY in your environment
# os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# Initialize Google Gemini model
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text:\n{text}",
    input_variables=["text"]
)

# # Run first prompt
# prompt1 = template1.invoke({"topic": "2023 Cricket World Cup in India"})
# result = chat.invoke(prompt1)

# # Run second prompt using the result of first
# prompt2 = template2.invoke({"text": result.content})
# result1 = chat.invoke(prompt2)

# # Final summary output
# print("\n--- Detailed Report ---\n")
# print(result.content)

# print("\n--- 5 Line Summary ---\n")
# print(result1.content)



parser=StrOutputParser()

chain = template1 | chat | parser | template2 | chat | parser
result = chain.invoke({"topic": "2023 Cricket World Cup in India"})
print(result)