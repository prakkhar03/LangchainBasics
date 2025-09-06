from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import  AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

#AI MESSAGE
#AIMessage is returned from a chat model as a response to a prompt.
#This message represents the output of the model and consists of both the raw output as returned by
# the model together standardized fields (e.g., tool calls, usage metadata) added by the LangChain framework.

#Human MESSAGE
#HumanMessage is used to represent input from a human user in a chat-based interaction.
# It encapsulates the text or content provided by the user and is typically used as input to
# a chat model or conversational agent.

#SYSTEM MESSAGE
#SystemMessage is used to represent system-level instructions or context in a chat-based interaction.
# It provides additional information or directives that can influence the behavior of the chat model or agent.


model= GoogleGenerativeAI(model="gemini-2.5-pro")
messages=[
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following English text to French: 'Hello, how are you?'")
]
result=model.invoke(messages)
messages.append(AIMessage(content=result))
print(messages)