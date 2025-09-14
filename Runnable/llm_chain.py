from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

load_dotenv()
model=GoogleGenerativeAI(model="gemini-2.5-pro")
prompt=PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)
chain=LLMChain(llm=model,prompt=prompt)
topic=input("Enter a topic")
output=chain.run(topic)
print(output)
