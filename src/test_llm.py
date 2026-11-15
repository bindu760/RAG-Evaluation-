
from llm_model import get_llm


# Initialize Groq LLM
llm = get_llm()

# Send a question to the model
response = llm.invoke(
    "Explain Artificial Intelligence in simple words."
)

# Display response
print("AI Response:")
print(response.content)
