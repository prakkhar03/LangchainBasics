from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
model = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0 
)
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
template2=PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)
parser=StrOutputParser()
# chain=template1 | model | parser  #simple chain
chain=template1 | model | parser | template2 | model | parser #sequential chain
result=chain.invoke({'topic':'football'})
print(result)
chain.get_graph().print_ascii()