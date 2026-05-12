import os

from langchain_google_genai import ChatGoogleGenerativeAI


THE_KEY = os.getenv("THE_KEY")

if not THE_KEY:
    raise ValueError("THE_KEY not found in environment variables")


LLM_MODEL = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=THE_KEY
)


def llm(smiles, predicted_solubility):

    prompt = f"""
You are a solubility interpretation assistant for drug discovery.

INPUT:
SMILES: {smiles}
Predicted logS (mol/L): {predicted_solubility}

...
"""

    response = LLM_MODEL.invoke(prompt)

    return response.content