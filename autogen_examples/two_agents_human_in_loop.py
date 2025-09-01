import asyncio
import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv


async def main():
    """
    Workflow:
        1. Loads environment variables from a `.env` file.
        2. Reads the `GOOGLE_API_KEY` required for the Gemini model.
        3. Creates an `AssistantAgent` (math tutor) powered by the Gemini API.
        4. Creates a `UserProxyAgent` that simulates a human user (student).
        5. Organizes agents into a round-robin chat team.
        6. Runs a console-based interactive tutoring session until the user
           (or proxy) mentions "LESSON COMPLETED".
    """
    try:
        # Load environment variables from .env file
        loaded = load_dotenv(".env", override=True)
        if not loaded:
            raise FileNotFoundError(".env file not found in project directory.")
    except Exception as e:
        print(f"Error loading environment file: {e}")
        return

    # Retrieve Google API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in .env file.")
        return

    # Initialize Gemini chat completion client
    gemini_model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=api_key,
    )

    # Create an mcp server
    fs_mcp_server_params = StdioServerParams(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/home/gizachew/main/ai/agentic_ai_experimentation/autogen_examples",
        ],
        read_timeout_secons=60,
    )

    # Start the mcp server
    fs_workbench = McpWorkbench(fs_mcp_server_params)

    async with fs_workbench as fs_wb:

        # Define assistant agent (math tutor) with system instructions
        agent = AssistantAgent(
            name="MathTutor",
            model_client=gemini_model_client,
            system_message=(
                "You are a helpful math tutor. Help the user solve math problems. "
                "Keep your responses short and clear. "
                "You have access to file system. Feel free to create"
                "files to help with student learning."
                "When the user says 'Thanks Done' or something similar, "
                "acknowledge and say 'LESSON COMPLETED' to end the session."
            ),
        )

        # Define a proxy agent that represents the student
        user_proxy = UserProxyAgent(name="Student")

        # Create a team chat where participants take turns (round-robin)
        team = RoundRobinGroupChat(
            participants=[user_proxy, agent],
            termination_condition=TextMentionTermination("LESSON COMPLETED"),
        )

        # Run the interactive tutoring session in the console
        await Console(team.run_stream(task="I need some help with multiplication."))

    await gemini_model_client.close()


if __name__ == "__main__":
    """
    Launch the asynchronous main() function using asyncio.
    Ensures the event loop is properly managed when running the script directly.
    """
    asyncio.run(main())
