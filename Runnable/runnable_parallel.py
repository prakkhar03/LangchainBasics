from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence,RunnableParallel

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
model = GoogleGenerativeAI(
    model="gemini-2.5-pro"
)
template1 = PromptTemplate(
    template="Write an Instagram post on  {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Write a linkedin post on  {topic}",
    input_variables=["topic"]
)
parser=StrOutputParser()
chain = RunnableParallel({
    "Insta": RunnableSequence(template1, model, parser),
    "Linkedin": RunnableSequence(template2, model, parser),
})
result=chain.invoke({'topic':'Django'})
print(result)