# from typing_extensions import Annotated
# from pydantic import BaseModel, EmailStr
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os

# load_dotenv()
# #PYDANTIC
# # Pydantic is a data validation and settings management library for Python, based on type annotations.
# # It allows you to define data models with type hints, and it automatically validates and parses input
# # data to ensure it conforms to the specified types and constraints.
# # Pydantic is widely used in FastAPI for request validation and response serialization.

# class Student(BaseModel):
#     name: str
#     age: int
#     email:EmailStr
#     is_active: bool = True
#     grades: Annotated[list[int], "List of grades the student has received"]


# student=Student(
#     name="John Doe",
#     age=20,
#     email="xyz@gmail.com",
#     grades=[85, 90, 78]
# )
# print(student)

#Benifits of Pydantic
#1. Data Validation: Pydantic automatically validates data based on the types and constraints defined
# in the model. This helps catch errors early and ensures data integrity.
#2. Type Safety: By using type annotations, Pydantic provides type safety, making
# it easier to understand the expected data types and catch type-related errors.
#3. Serialization and Deserialization: Pydantic models can be easily serialized to and deserialized from
# various formats, such as JSON, making it convenient for data exchange.
#4. Integration with FastAPI: Pydantic is tightly integrated with FastAPI, allowing
# for seamless request validation and response serialization in web applications.
from typing import Literal, Optional
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Schema
class Review(BaseModel):
    key_themes: list[str]
    summary: str
    sentiment: Literal["pos", "neg"]
    pros: Optional[list[str]]
    cons: Optional[list[str]]
    name: Optional[str]

# Wrap with structured output
structured_model = model.with_structured_output(Review)

# Invoke
result = structured_model.invoke("""
The 2023 Cricket World Cup in India was a rollercoaster of emotions! 
From packed stadiums with electrifying crowds to some of the best batting 
and bowling performances in recent history, it truly showcased why cricket 
is called a religion in India. The pitches offered a fair balance—high-scoring 
thrillers in some venues and bowler-friendly tracks in others, keeping fans 
on the edge of their seats.

The highlight for me was India’s dominating run in the group stages—Virat 
Kohli’s consistency and Mohammed Shami’s lethal spells were unforgettable. 
Australia, as always, proved why they are big-match players, snatching the cup 
with a clinical performance in the final.

However, the tournament wasn’t perfect. Scheduling hiccups, delayed ticket sales, 
and uneven pitch reports did raise eyebrows. Also, some matches felt too one-sided, 
which slightly killed the suspense.

Pros:
- Electrifying performances (Kohli, Shami, Maxwell’s 201*, etc.)
- Fantastic crowd energy and stadium atmosphere
- Balanced pitches offering variety (high scores + bowler-friendly games)
- A record-breaking edition in terms of viewership and stats

Cons:
- Scheduling and ticketing chaos frustrated fans
- Some mismatches lacked competitiveness
- Home crowd expectations led to extra pressure on India

Review by Prakhar
""")

print(result)
