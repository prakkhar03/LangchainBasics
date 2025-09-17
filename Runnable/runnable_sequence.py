from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
model = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0 
)
template1 = PromptTemplate(
    template="Write a 6 line report on {topic}",
    input_variables=["topic"]
)
template2=PromptTemplate(
    template='Generate a 1 pointer summary from the following text \n {text}',
    input_variables=['text']
)
parser=StrOutputParser()
# chain=template1|model|parser
chain=RunnableSequence(template1,model,parser,template2,model,parser)
result=chain.invoke({'topic':'AI'})
print(result)