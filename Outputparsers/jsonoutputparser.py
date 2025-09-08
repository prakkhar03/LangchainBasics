from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=GOOGLE_API_KEY   
)

parser = JsonOutputParser()

# Prompt with format instructions
template = PromptTemplate(
    template=(
        "Write a detailed report on {topic}.\n\n"
        "Return following in this format:\n{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"topic": "2023 Cricket World Cup in India"})
print(result) 
