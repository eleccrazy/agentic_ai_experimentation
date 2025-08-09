"""
File: groq_llm.py
Description: This module contains the code for experimenting llama3-8b-8192 model using Groq's API.
Author: Gizachew Kassa
Date Created: 09/08/2025
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# load environment variables from .env file for API keys and configurations
load_dotenv()

llm = ChatGroq(model="llama3-8b-8192")
response = llm.invoke("Explain vector databases in one paragraph.")

# The text you want is in the `.content` attribute:
print(response.content)
