from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st # type: ignore
from langchain.prompts import PromptTemplate, load_prompt
import os
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
#Static Prompt
# st.header("Research Assistant")
# input=st.text_input("Enter your query:", key="query")
# if st.button("Get Answer"):
#     model = GoogleGenerativeAI(model="gemini-1.5-pro")
#     result=model.invoke(input)
#     st.write(result)
#Prompt Template
# A PromptTemplate in LangChain is a structured way to create prompts
# dynamically by inserting variables into a predefined template. Instead of
# hardcoding prompts, PromptTemplate allows you to define placeholders that
# can be filled in at runtime with different inputs.
# This makes it reusable, flexible, and easy to manage, especially when working
# with dynamic user inputs or automated workflows.

#Dynamic Prompt
st.header("Research Assistant")
input=st.text_input("Enter your query:", key="query")

model= GoogleGenerativeAI(model="gemini-1.5-pro")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

# prompt=template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
# })
# if st.button("Summarize"):
#     result=model.invoke(prompt)
#     st.write(result.content)

##rather than invoking model separately we can create a chain


if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)