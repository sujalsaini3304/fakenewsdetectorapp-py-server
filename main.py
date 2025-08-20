from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the OpenAI client with custom base URL and API key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

@tool
def ddg_search(query: str, max_results: int = 5):
    """Search DuckDuckGo for real-time information."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results

# Creating AI Agent
agent = create_react_agent(
    model=ChatOpenAI(
        model="qwen/qwen3-235b-a22b:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    ),
    tools=[ddg_search],
    prompt="""
    You are an AI Fake News Detector.
    Analyze any text, headline, or claim and respond ONLY in JSON format with these fields:
    {
    "claim": "...",
    "verdict": "True | Misleading | Fake",
    "reasoning": "short explanation with sources if possible"
    }
    Stay neutral, objective, and educational.
    """
)

class UserTextualData(BaseModel):
    text: str

# The agent endpoint
@app.post("/api/validate")
async def RAG_Agent(userPrompt: UserTextualData) -> dict:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": userPrompt.text}]}
    )

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if msg.type == "ai":
            content = msg.content.strip()

            # Extract JSON block using regex
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group())
                    return {"message": "Data verified successfully.", "data": data}
                except json.JSONDecodeError:
                    return {"message": "Error parsing AI response JSON.", "data": None}

            return {"message": "No valid JSON found.", "data": None}

    return {"message": "No conclusion found.", "data": None}

@app.get("/")
def root():
    return {"message": "Server is running."}
