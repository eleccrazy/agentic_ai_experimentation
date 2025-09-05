"""
File: output_formatting.py
Description:
    This module demonstrates multiple techniques for handling and formatting
    outputs from Large Language Models (LLMs) in the context of prompt engineering.

    Specifically, it shows how to:
        1. Use LLMs without structured output.
        2. Use prompting strategies to enforce structured output.
        3. Use LangChain's PydanticOutputParser for structured parsing.
        4. Leverage model-native structured output (where supported).

Author: Gizachew Kassa
Date Created: 05/09/2025
"""

import os

from dotenv import load_dotenv
from langchain.output_parsers.pydantic import PydanticOutputParser
from llms import get_llm
from paths import APP_CONFIG_FPATH, OUTPUTS_DIR
from pydantic import BaseModel, Field
from utils import load_publication, load_yaml_config, save_text_to_file

# Load environment variables (e.g., API keys)
load_dotenv()


# Pydantic Models for Structured Data
class Entity(BaseModel):
    """
    Represents an individual entity mentioned in the publication.

    Attributes:
        type (str): The type of entity. Must be either "model" or "task".
        name (str): The name of the entity (e.g., "GPT-4", "Text Classification").
    """

    type: str = Field(description="The type of the entity. Either 'model' or 'task'")
    name: str = Field(description="The name of the entity")


class Entities(BaseModel):
    """
    Represents a collection of entities extracted from a publication.

    Attributes:
        entities (list[Entity]): A list of extracted entities.
    """

    entities: list[Entity] = Field(
        description="The entities mentioned in the publication"
    )


# Example Techniques


def no_structured_output(model: str = "gpt-4o-mini") -> None:
    """
    Demonstrates basic LLM usage without structured output.

    Steps:
        1. Load publication content.
        2. Create a natural language prompt asking the LLM to extract entities.
        3. Invoke the LLM.
        4. Save the raw response (unstructured text) to a file.

    Args:
        model (str): The LLM model name to use. Defaults to "gpt-4o-mini".
    """
    publication_content = load_publication()

    prompt = f"""
    Provide a list of entities mentioned in the publication. 
    An entity is either a model or a task.

    <publication>
    {publication_content}
    </publication>
    """

    llm = get_llm(model)
    response = llm.invoke(prompt)

    # Save both the prompt and response for analysis
    saved_text = f"""# Prompt: {prompt}

# Response:
{response.content}
    """

    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR, "no_structured_output_llm_response.md"),
        header="LLM Response Without Structured Output",
    )


def with_prompting_to_structure_output(model: str = "gpt-4o-mini") -> None:
    """
    Demonstrates using explicit prompting to enforce structured output.

    Technique:
        Instead of letting the LLM respond freely, we instruct it to
        return data in JSON format with specific fields.

    Args:
        model (str): The LLM model name to use. Defaults to "gpt-4o-mini".
    """
    publication_content = load_publication()

    prompt = f"""
    Provide a list of entities mentioned in the publication. 
    An entity is either a model or a task.

    <publication>
    {publication_content}
    </publication>

    Return a JSON object with a single field "entities" which is a list of dictionaries. 
    Each dictionary should have two fields: "type" and "name".

    Example:
    {{
        "entities": [
            {{
                "type": "model",
                "name": "GPT-4"
            }},
            {{
                "type": "task",
                "name": "Text Classification"
            }}
        ]
    }}
    """

    llm = get_llm(model)
    response = llm.invoke(prompt)

    saved_text = f"""# Prompt: {prompt}

# Response:
{response.content}
    """

    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR, "with_prompting_to_structure_output_llm_response.md"),
        header="LLM Response With Prompting to Structure Output",
    )


def with_output_parser(model: str = "gpt-4o-mini") -> None:
    """
    Demonstrates structured parsing using LangChain's PydanticOutputParser.

    Technique:
        1. Define a Pydantic model (Entities) for expected output.
        2. Retrieve parser-specific formatting instructions.
        3. Include these instructions in the prompt.
        4. Parse the LLM’s raw response into a validated Pydantic object.

    Args:
        model (str): The LLM model name to use. Defaults to "gpt-4o-mini".
    """
    publication_content = load_publication()

    prompt_template = """
    Provide a list of entities mentioned in the publication. 
    An entity is either a model or a task.

    <publication>
    {publication_content}
    </publication>

    {format_instructions}
    """

    llm = get_llm(model)

    # Create parser for the Entities schema
    output_parser = PydanticOutputParser(pydantic_object=Entities)
    format_instructions = output_parser.get_format_instructions()

    # Fill in the prompt template with actual data and formatting rules
    prompt = prompt_template.format(
        publication_content=publication_content,
        format_instructions=format_instructions,
    )

    response = llm.invoke(prompt)

    # Attempt to parse into Pydantic model
    parsed_response = output_parser.parse(response.content)

    saved_text = f"""# Prompt: {prompt}

# Before Parsing:
{response.content}

# After Parsing:
{parsed_response}
    """

    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR, "with_output_parser_llm_response.md"),
        header="With Output Parser",
    )


def model_native_structured_output(model: str = "gpt-4o-mini") -> None:
    """
    Demonstrates leveraging the model’s native structured output capabilities
    (if supported by the LLM provider).

    Technique:
        Instead of relying on prompt instructions or parsers,
        the model itself directly returns structured data that maps
        to the Pydantic model.

    Args:
        model (str): The LLM model name to use. Defaults to "gpt-4o-mini".
    """
    publication_content = load_publication()

    prompt = f"""
    Provide a list of entities mentioned in the publication. 
    An entity is either a model or a task.

    <publication>
    {publication_content}
    </publication>
    """

    # Request model-native structured output aligned to Entities schema
    llm = get_llm(model).with_structured_output(Entities)
    response = llm.invoke(prompt)

    saved_text = f"""## Prompt: {prompt}

## Response:
{str(response.model_dump())}
    """

    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR, "model_native_structured_output_llm_response.md"),
        header="LLM Response With Model Native Structured Output",
    )


# Entry Point
if __name__ == "__main__":
    """
    Entry point for running the script.

    Loads configuration (e.g., which LLM model to use)
    and executes one of the demonstration functions.
    """
    config = load_yaml_config(APP_CONFIG_FPATH)
    model = config.get(
        "llm", "gpt-4o-mini"
    )  # I used 'gemini-1.5-flash'. You can set this from the config.yaml file of your llm choice.

    # Uncomment functions below to run examples

    # no_structured_output(model)
    # with_prompting_to_structure_output(model)
    # with_output_parser(model)
    model_native_structured_output(model)
