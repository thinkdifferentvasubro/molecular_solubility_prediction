from langchain_google_genai import ChatGoogleGenerativeAI
import os
def llm(smiles, predicted_solubility):
    the_key = os.getenv("THE_KEY")
    if not the_key:
        raise ValueError("THE_KEY not found in environment variables")
    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=1.0, 
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=the_key
        )
    promt = f"""
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
- Do NOT infer structure from SMILES (they are labels only)
- Do NOT assume dose, permeability, metabolism, or experiments
- Avoid repetition across sections
- Use probabilistic language (e.g., likely, may, could)
- Keep output concise and information-dense

Validation:
- If length mismatch → return: "Error: SMILES and logS length mismatch."
- If invalid logS → return: "Error: Invalid logS input."

---

6-Class LogS Binning:
Class 1: logS < -4.7180 → Extremely low  
Class 2: -4.7180 to -3.6239 → Very low  
Class 3: -3.6239 to -2.6185 → Low  
Class 4: -2.6185 to -1.9030 → Moderate  
Class 5: -1.9030 to -0.8508 → Good  
Class 6: ≥ -0.8508 → High  

---

TASK:
1. Detect input type
2. For each (SMILES, logS):

Generate a **concise, non-redundant interpretation** including:

- Solubility class + 1-line reasoning
- Practical implication (merge dissolution, absorption, formulation into ONE insight)
- Chemical interpretation (polarity vs lipophilicity)
- Optimization need (yes/no + why)
- Final summary (2–3 sentences, decision-oriented, no repetition)

---

OUTPUT FORMAT:

IF SINGLE INPUT:

SMILES: (smiles) 
logS: (value) → (class)

Reasoning:  
(one short sentence explaining classification)

Practical implication:  
(one compact insight covering dissolution + absorption + formulation)

Chemical interpretation:  
(what solubility suggests about polarity/lipophilicity)

Optimization:  
(Needed / Not needed + brief reason)

Summary:  
(2–3 sentences, concise, no repetition)

---

IF LIST INPUT:

For each pair:

SMILES: (smiles_i)  
logS: (value) → (class)

Reasoning:  
...

Practical implication:  
...

Chemical interpretation:  
...

Optimization:  
...

Summary:  
..."""

    response = model.invoke(promt)
    print(type(response.content))
    return response.content[0]["text"]
