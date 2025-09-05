from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
#GEMINI
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# result = model.invoke('What is the capital of India')

# print(result.content)

#OPENAI
# from langchain.chat_models import ChatOpenAI
# model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
#temprature means creativity of the model
# max_completion_tokens means maximum length of the response
# result = model.invoke("Write a 5 line poem on cricket")

# print(result.content)

#HUGGINGFACE
# from langchain_huggingface import HuggingFaceHub
# model = HuggingFaceHub(repo_id="google/flan-t5-xxl",
#                         model_kwargs={"temperature":1.5, "max_new_tokens":100})
# result = model.invoke("Write a 5 line poem on cricket")
# print(result)

#Claude
# from langchain_claude import ChatClaude
# model = ChatClaude(model="claude-2", max_retries=3, temperature
#                     , max_tokens_to_sample=1000)
# result = model.invoke("Write a 5 line poem on cricket")
# print(result.content)
# max_retries means if the model fails to respond it will retry 3 times


from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
#pipeline_kwargs are the parameters for the model
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)