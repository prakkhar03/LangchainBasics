from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# result = model.embed_query('What is the capital of India')
documents = ['What is the capital of India', 'What is the capital of USA', 'What is the capital of UK', 'What is the capital of Australia']
result = model.embed_documents(documents)
print(result)