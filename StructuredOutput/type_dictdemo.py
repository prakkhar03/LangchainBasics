# from typing import TypedDict    
# class Movie(TypedDict):
#     name: str
#     year: int
#     rating: float


# movie: Movie = {
#     'name': 'Inception',
#     'year': 2010,
#     'rating': 8.8
# }
# print(movie)

# from langchain_openai import ChatOpenAI # type: ignore
# from typing import TypedDict

# class ResponseSchema(TypedDict):
#     title: str
#     review: str
#     sentiment: str

# llm = ChatOpenAI(model="gpt-4o-mini")
# structured = llm.with_structured_output(ResponseSchema)
# print(structured.invoke("Provide a movie review for Inception"))


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
import os
load_dotenv()
OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')

model = ChatOpenAI()

# schema
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The 2023 Cricket World Cup in India was a rollercoaster of emotions! From packed stadiums with electrifying crowds to some of the best batting and bowling performances in recent history, it truly showcased why cricket is called a religion in India. The pitches offered a fair balance—high-scoring thrillers in some venues and bowler-friendly tracks in others, keeping fans on the edge of their seats.

The highlight for me was India’s dominating run in the group stages—Virat Kohli’s consistency and Mohammed Shami’s lethal spells were unforgettable. Australia, as always, proved why they are big-match players, snatching the cup with a clinical performance in the final.

However, the tournament wasn’t perfect. Scheduling hiccups, delayed ticket sales, and uneven pitch reports did raise eyebrows. Also, some matches felt too one-sided, which slightly killed the suspense.

Pros:

Electrifying performances (Kohli, Shami, Maxwell’s 201*, etc.)

Fantastic crowd energy and stadium atmosphere

Balanced pitches offering variety (high scores + bowler-friendly games)

A record-breaking edition in terms of viewership and stats

Cons:

Scheduling and ticketing chaos frustrated fans

Some mismatches lacked competitiveness

Home crowd expectations led to extra pressure on India

Review by Prakhar""")

print(result['name'])