import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


async def main():
    """
    Run a short, round-robin discussion between two systems-engineering agents about
    building automation with agentic AI.

    Workflow:
      1) Environment bootstrap:
         - Load variables from a local `.env` file.
         - Expect `GOOGLE_API_KEY`; exit with a clear message if `.env` or the key is missing.
      2) Model client:
         - Initialize an OpenAI-compatible client targeting `gemini-1.5-flash-8b`.
      3) Agents:
         - `senior_se`: A senior systems engineer mentoring and guiding design decisions,
                        offering concrete architecture advice and guardrails.
         - `newhire_se`: A new systems engineer with no prior agentic-AI experience,
                         asking clarifying questions and surfacing uncertainties.
      4) Orchestration:
         - Use `RoundRobinGroupChat` so the two agents alternate.
         - Stop after a bounded number of messages via `MaxMessageTermination`.
      5) Output:
         - Stream the conversation to the console UI.

    This example demonstrates a lightweight agent-to-agent knowledge-transfer scenario
    focused on agentic-AI automation patterns (task decomposition, tool use, observability,
    safety/guardrails, and deployment considerations).
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

    # Senior systems engineer: explains, mentors, proposes architecture & guardrails
    agent_senior = AssistantAgent(
        name="senior_se",
        model_client=gemini_model_client,
        system_message=(
            "You are a senior systems engineer mentoring a new hire on automation with agentic AI. "
            "Explain concepts clearly, propose concrete architectures (task orchestrator, tools, memory, "
            "retrieval, eval/observability), discuss risks (hallucinations, cost, latency, PII), and "
            "offer actionable steps (MVP scope, metrics, rollout plan). Ask occasional check-questions to "
            "confirm understanding."
        ),
    )

    # New hire: asks basic questions, surfaces unknowns, requests examples
    agent_newhire = AssistantAgent(
        name="newhire_se",
        model_client=gemini_model_client,
        system_message=(
            "You are a new systems engineer with no prior experience in agentic-AI automation. "
            "Be curious and honest about what you don't know. Ask simple, clarifying questions, "
            "request concrete examples (APIs, tools, data flows), and summarize what you learn."
        ),
    )

    team = RoundRobinGroupChat(
        participants=[agent_senior, agent_newhire],
        termination_condition=MaxMessageTermination(max_messages=8),
    )

    # Initial discussion prompt for the two SE agents
    kickoff_task = (
        "We need to automate Level-1 on-call runbook tasks (log triage, service restarts, ticket updates) "
        "using agentic AI. Please discuss a pragmatic MVP: required tools/APIs, orchestration approach, "
        "risk mitigations, observability/metrics, and a safe rollout plan. Assume the new hire is unfamiliar "
        "with agentic-AI concepts."
    )

    # Stream the conversation to the console
    await Console(team.run_stream(task=kickoff_task))


if __name__ == "__main__":
    asyncio.run(main())
