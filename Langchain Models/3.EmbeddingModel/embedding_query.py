from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
#GEMINI
model = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , dimension=1024)
result = model.embed_query('What is the capital of India')
print(result)
#embed_query is used to get the embedding of a single query
#dimension is the size of the embedding vector