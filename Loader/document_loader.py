from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))


# #PDF loader
# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader('dl-curriculum.pdf')

# docs = loader.load()

# print(len(docs))

# print(docs[0].page_content)
# print(docs[1].metadata)


#CSV Loader
# from langchain_community.document_loaders import CSVLoader

# loader = CSVLoader(file_path='Social_Network_Ads.csv')

# docs = loader.load()

# print(len(docs))
# print(docs[1])

#Directory loader
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# loader = DirectoryLoader(
#     path='books',
#     glob='*.pdf',
#     loader_cls=PyPDFLoader
# )

# docs = loader.lazy_load()

# for document in docs:
#     print(document.metadata)


#Webbased loader
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI()

# prompt = PromptTemplate(
#     template='Answer the following question \n {question} from the following text - \n {text}',
#     input_variables=['question','text']
# )

# parser = StrOutputParser()

# url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
# loader = WebBaseLoader(url)

# docs = loader.load()


# chain = prompt | model | parser

# print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))