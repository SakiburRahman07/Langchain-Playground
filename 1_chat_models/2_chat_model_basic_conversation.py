# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Create a ChatGroq model
model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define a message
messages = [
    (
        "system",
        "You are a helpful assistant that tell about bangladesh.",
    ),
    ("human", "what is the national language of bangladesh"),
]

# Invoke the model with the messages
result = model.invoke(messages)

# Print the result
print("Full result:")
print(result.content)
