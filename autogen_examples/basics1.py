import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


async def main():
    """
    Entry point for running the Gemini-powered AssistantAgent.

    This function:
    1. Loads environment variables from a `.env` file (expects GOOGLE_API_KEY).
       - Raises a FileNotFoundError if `.env` is missing.
       - Exits gracefully if GOOGLE_API_KEY is not set.
    2. Initializes an OpenAI-compatible client for Google's Gemini model.
    3. Creates an AssistantAgent bound to that model client.
    4. Runs the agent with a sample task ("what is 25 * 4?") and streams
       the response to the console.

    Intended as a minimal working example of using AutoGen AgentChat with
    Gemini via an OpenAI-compatible interface.
    """
    try:
        loaded = load_dotenv(".env", override=True)
        if not loaded:
            raise FileNotFoundError(".env file not found in project directory.")
    except Exception as e:
        print(f"Error loading environment file: {e}")
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in .env file.")
        return

    gemini_model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=api_key,
    )
    gemni_agent = AssistantAgent(name="assistant", model_client=gemini_model_client)

    await Console(gemni_agent.run_stream(task="what is 25 * 3?"))


if __name__ == "__main__":
    asyncio.run(main())
