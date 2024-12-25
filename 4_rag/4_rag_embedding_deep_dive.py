import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma


# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

# Read the text content from the file
try:
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
except Exception as e:
    raise RuntimeError(f"Error reading the file {file_path}: {e}")

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")


# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")


# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        try:
            db = Chroma(
                persist_directory=persistent_directory,
                embedding_function=embedding_function,
            )
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.1},
            )
            relevant_docs = retriever.invoke(query)
            # Display the relevant results with metadata
            if relevant_docs:
                print(f"\n--- Relevant Documents for {store_name} ---")
                for i, doc in enumerate(relevant_docs, 1):
                    print(f"Document {i}:\n{doc.page_content}\n")
                    if doc.metadata:
                        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
            else:
                print("No relevant documents found.")
        except Exception as e:
            print(f"Error querying the vector store {store_name}: {e}")
    else:
        print(f"Vector store {store_name} does not exist.")


# Create vector store using Hugging Face Embeddings
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

# Define the user's question
query = "Who is Odysseus' wife?"

# Query the vector store
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

print("Querying demonstrations completed.")
