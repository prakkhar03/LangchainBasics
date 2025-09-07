from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
import os
import json
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)
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

print(result)