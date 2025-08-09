"""
File: gemini_llm.py
Description: This module contains the code for experimenting with Google's Gemini API for language models.
Author: Gizachew Kassa
Date Created: 10/08/2025
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize the Google Gemini LLM with specific parameters
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
response = llm.invoke("Explain prompt engineering in simple terms.")
print(response.content)
