from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

model = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , dimension=1024)
documents = ['Delhi is the capital of India', 'Washington D.C. is the capital of USA', 'London is the capital of UK', 'Canberra is the capital of Australia']
query = 'What is the capital of India'
doc_embeddings = model.embed_documents(documents)
query_embedding = model.embed_query(query)
#calculate similarity between query and documents
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
similarities = cosine_similarity([query_embedding], doc_embeddings)
print(similarities)
# get the index of the most similar document
most_similar_document_index = np.argmax(similarities)
print(f"The most similar document to the query '{query}' is: '{documents[most_similar_document_index]}' with similarity score of {similarities[0][most_similar_document_index]}")
#Cosine similarity is used to calculate the similarity between two vectors  