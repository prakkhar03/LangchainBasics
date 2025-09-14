from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser,SimpleJsonOutputParser,PydanticOutputParser
from  pydantic import BaseModel
import os
load_dotenv()

model=GoogleGenerativeAI(model="gemini-2.5-pro")
print(model)
prompt=PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)
topic=input("Enter a topic")

formatting_prompt=prompt.format(topic=topic)
blog_title=model.predict(formatting_prompt)
print("Generated_title:",blog_title)