from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
#GEMINI
model = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , dimension=1024)
documents = ['What is the capital of India', 'What is the capital of USA', 'What is the capital of UK', 'What is the capital of Australia']
result = model.embed_documents(documents)
print(result)
#embed_documents is used to get the embedding of multiple documents
#dimension is the size of the embedding vector