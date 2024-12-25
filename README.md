# LangChain Playground

Welcome to the LangChain Playground! This repository provides all the resources you need to explore LangChain's capabilities, from creating AI agents to building RAG chatbots and automating tasks with cutting-edge AI technology.

## Course Outline

1. **Setup Environment**
2. **Chat Models**
3. **Prompt Templates**
4. **Chains**
5. **RAG (Retrieval-Augmented Generation)**
6. **Agents & Tools**

## Getting Started

### Prerequisites

- **Python**: Version 3.10 or 3.11
- **Poetry**: Install Poetry by following [this guide](https://python-poetry.org/docs/#installation).

### Installation

1. **Clone the repository**:

   ```bash
   git clone repo-link
   cd langchain-crash-course
   ```

2. **Install dependencies using Poetry**:

   ```bash
   poetry install --no-root
   ```

3. **Set up environment variables**:

   Rename the `.env.example` file to `.env` and update it with your credentials:

   ```bash
   mv .env.example .env
   ```

4. **Activate the Poetry shell**:

   ```bash
   poetry shell
   ```

5. **Run code examples**:

   Execute the Python scripts to see LangChain in action:

   ```bash
   python 1_chat_models/1_chat_model_basic.py
   ```