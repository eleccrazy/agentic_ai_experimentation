"""
File: llms.py
Description: This module provides functions to get language model instances based on specified models.
Author: Gizachew Kassa
Date Created: 14/08/2025
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()


available_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "llama3-8b-8192",
]


def get_llm(model: str):
    if model not in available_models:
        raise ValueError(f"Invalid model. Available models: {available_models.keys()}")

    if model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    elif model == "llama3-8b-8192":
        return ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
