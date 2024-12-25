# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Initialize the ChatGroq model
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define messages for the model
messages = [
    (
        "system",
        "capital city",
    ),
    ("human", "what is the capital of bangladesh"),
]

# Invoke the model with the messages
ai_msg = llm.invoke(messages)

# Display the AI response
print(ai_msg.content)
