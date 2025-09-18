from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
def word_count(text):
    return len(text)
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
model = GoogleGenerativeAI(
    model="gemini-2.5-pro"
)
template = PromptTemplate(
    template="Write an Instagram post on  {topic}",
    input_variables=["topic"]
)
parser=StrOutputParser()
chain=RunnableSequence(template,model,parser)
parallel_chain=RunnableParallel({
    'post':chain,
    'word_count':RunnableLambda(word_count)
})
final_chain=RunnableSequence(chain,parallel_chain)
# result=final_chain.invoke({'topic':'AI'})
# print(result)
final_chain.get_graph().print_ascii()