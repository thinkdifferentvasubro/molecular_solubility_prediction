import os

from langchain_google_genai import ChatGoogleGenerativeAI


THE_KEY = os.getenv("THE_KEY")

if not THE_KEY:
    raise ValueError("THE_KEY not found in environment variables")


LLM_MODEL = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    google_api_key=THE_KEY
)


def llm(smiles, predicted_solubility):
    
    prompt = f"""
You are a solubility interpretation assistant for drug discovery.

INPUT:
SMILES: {smiles}
Predicted logS (mol/L): {predicted_solubility}

Input types:
- Single SMILES + single logS
- List of SMILES + list of logS

Alignment rule:
- If lists are provided, lengths MUST match
- Each SMILES corresponds to logS at the same index

---

STRICT RULES:
- Use ONLY the provided logS value(s)
- Do NOT infer molecular structure from SMILES
- Treat SMILES purely as identifiers/labels
- Do NOT assume dose, permeability, metabolism, toxicity, or experimental outcomes
- Avoid repetition across sections
- Use concise scientific language
- Use probabilistic wording where appropriate:
  "may", "could", "likely", "suggests"

Validation:
- If length mismatch → return exactly:
  "Error: SMILES and logS length mismatch."

- If invalid logS input → return exactly:
  "Error: Invalid logS input."

---

6-Class LogS Binning:

Class 1:
logS < -4.7180
→ Extremely low solubility

Class 2:
-4.7180 to -3.6239
→ Very low solubility

Class 3:
-3.6239 to -2.6185
→ Low solubility

Class 4:
-2.6185 to -1.9030
→ Moderate solubility

Class 5:
-1.9030 to -0.8508
→ Good solubility

Class 6:
≥ -0.8508
→ High solubility

---

TASK:

1. Detect whether input is:
   - single compound
   - multiple compounds

2. For EACH (SMILES, logS) pair generate:

- Solubility class
- One-line reasoning
- One compact practical implication
  (combine dissolution, absorption,
   and formulation insight into ONE statement)
- Chemical interpretation
  (polarity vs lipophilicity implications only)
- Optimization requirement
  (Needed / Not needed + short reason)
- Final concise decision-oriented summary

---

OUTPUT FORMAT

IF SINGLE INPUT:

SMILES: <smiles>

logS: <value>
Class: <class>

Reasoning:
<one concise sentence>

Practical implication:
<single compact insight>

Chemical interpretation:
<concise interpretation>

Optimization:
<Needed / Not needed + reason>

Summary:
<2-3 concise non-redundant sentences>

---

IF MULTIPLE INPUTS:

For EACH pair:

SMILES: <smiles>

logS: <value>
Class: <class>

Reasoning:
...

Practical implication:
...

Chemical interpretation:
...

Optimization:
...

Summary:
...

Separate compounds using:
--------------------------------------------------
"""
    response = LLM_MODEL.invoke(prompt)

    return response.content