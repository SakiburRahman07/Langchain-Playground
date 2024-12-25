# Example Source: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/

from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_groq import ChatGroq

"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=your-project-id
"""

# Load environment variables
load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "langchain-f1552"  # Replace with your Firebase project ID
SESSION_ID = "user_session_1"  # Use a unique session ID for each user
COLLECTION_NAME = "chat_history"  # Firestore collection for storing chat history

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")

class CustomFirestoreChatMessageHistory(FirestoreChatMessageHistory):
    """Custom FirestoreChatMessageHistory to ensure only string data is stored."""
    def add_user_message(self, message):
        # Cast message content to string before storing
        super().add_user_message(str(message))

    def add_ai_message(self, message):
        # Cast AI message content to string before storing
        super().add_ai_message(str(message))


chat_history = CustomFirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize Chat Model
print("Initializing Groq Model...")
model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
print("Groq Model Initialized.")

print("Start chatting with the AI. Type 'exit' to quit.")

# Chat loop
while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    # Add user input to Firestore chat history
    chat_history.add_user_message(human_input)

    # Get AI response based on the chat history
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)  # Save AI response to Firestore

    # Print AI response
    print(f"AI: {ai_response.content}")

# Display final chat history
print("---- Final Chat History ----")
print(chat_history.messages)
