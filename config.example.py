"""
Configuration file for API keys

For local development:
1. Copy this file to config.py
2. Fill in your API keys

For Streamlit Cloud:
- Add API keys in Secrets section (not in code)
- This file reads from environment variables automatically
"""

import os

# OpenAI API Key
# For Streamlit Cloud: set in Secrets (OPENAI_API_KEY = "sk-proj-...")
# For local: uncomment line below and add your key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# OPENAI_API_KEY = "sk-proj-your-api-key-here"  # Uncomment for local development
