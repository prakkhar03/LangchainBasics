from  langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI

loader=TextLoader("docs.txt")
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunks_size=500 , chunks_overlap=50)
docs=text_splitter.split_documents(documents)

vectorstore=FAISS.from_document(docs,OpenAIEmbeddings)

retriever=vectorstore.as_retriever()

query="What are the key takeaways from the document?"
retrievered_docs=retriever.get_relevent_documents(query)
retrievered_text="\n".join([docs.page_content for docs in retrievered_docs])

model=GoogleGenerativeAI(model="gemini-2.5-pro")

prompt=f"Based on the following text, answer the question :{query} \n \n {retrievered_text}"
answer=model.predict(prompt)
print(answer)