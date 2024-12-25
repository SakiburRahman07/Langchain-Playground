from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
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

# Define prompt templates
feature_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)

pros_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        (
            "human",
            "Given these features: {features}, list the pros of these features.",
        ),
    ]
)

cons_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        (
            "human",
            "Given these features: {features}, list the cons of these features.",
        ),
    ]
)

# Define analysis branches
pros_branch_chain = (
    RunnableLambda(lambda x: pros_template.format_prompt(features=x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: cons_template.format_prompt(features=x)) | model | StrOutputParser()
)

# Combine pros and cons into a final review
combine_pros_cons = RunnableLambda(
    lambda x: f"Pros:\n{x['branches']['pros']}\n\nCons:\n{x['branches']['cons']}"
)

# Create the combined chain
chain = (
    feature_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | combine_pros_cons
)

# Run the chain
result = chain.invoke({"product_name": "Nokia 3310"})

# Output
print(result)
