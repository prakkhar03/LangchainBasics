# pdf_chatbot.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


load_dotenv()  
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Please set your GOOGLE_API_KEY in the .env file or environment variables.")


def load_pdf_documents(folder_path: str):
    
    docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(folder_path, pdf_file))
        docs.extend(loader.load())
    
    return docs

def create_vector_store(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store


def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = GoogleGenerativeAI(google_api_key=API_KEY)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def main():
    st.title("PDF Chatbot - RAG App")

    folder_path = st.text_input("Enter path to your PDF folder:", value="pdf_docs")

    if st.button("Load PDFs"):
        if not os.path.exists(folder_path):
            st.error(f"Folder {folder_path} does not exist!")
            return

        with st.spinner("Loading PDFs..."):
            docs = load_pdf_documents(folder_path)
            if not docs:
                st.error("No PDF documents found in folder!")
                return
            st.success(f"Loaded {len(docs)} PDF documents.")

            # Create FAISS index
            vector_store = create_vector_store(docs)
            st.success("Vector store created successfully!")

            # Build RAG chain
            qa_chain = build_rag_chain(vector_store)
            st.session_state['qa_chain'] = qa_chain
            st.success("Chatbot is ready!")

    if 'qa_chain' in st.session_state:
        query = st.text_input("Ask a question about your PDFs:")
        if query:
            with st.spinner("Fetching answer..."):
                result = st.session_state['qa_chain'].run(query)
                st.write(result)

if __name__ == "__main__":
    main()
