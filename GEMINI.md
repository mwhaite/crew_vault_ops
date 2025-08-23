# GEMINI.md - Vault Ops Project

## Project Overview

This project is a Python-based tool that provides a unified interface for using an Obsidian vault as a Retrieval-Augmented-Generation (RAG) knowledge base. It's designed to be used with `crewai` and allows for natural language interaction with the vault.

The core technologies used are:
- **Python:** The main programming language.
- **CrewAI:** For creating and managing AI agents.
- **Sentence-Transformers:** For generating text embeddings.
- **FAISS:** For efficient similarity search of vector embeddings.
- **Pydantic:** For data validation.

The project is structured into three main components:
- `vault_ops/vault_ops_tool.py`: The main `crewai` tool that orchestrates the other components.
- `vault_ops/embedder.py`: Handles text embedding and FAISS index management.
- `vault_ops/maintenance.py`: Provides vault maintenance utilities.

## Building and Running

### Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

### Running the Tool

The tool is intended to be used as a `crewai` tool. The following is an example of how to use it directly:

```python
from vault_ops.vault_ops_tool import VaultOpsTool

# Initialize the tool
tool = VaultOpsTool()

# Example usage:
# Create a new note
print(tool._run(action="create", path="test.md", text="Hello world!"))

# Ask a question about the vault's content
print(tool._run(action="ask", text="What is this about?"))

# Run vault maintenance
print(tool._run(action="maintenance"))

# Delete a note
print(tool._run(action="delete", path="test.md"))
```

**TODO:** Add a more comprehensive example of how to integrate this tool with a `crewai` agent.

## Development Conventions

### Code Style

The code follows the standard PEP 8 style guide for Python.

### Testing

There are no tests in the project yet.

**TODO:** Add a test suite to ensure the tool is working as expected.

### Contribution Guidelines

There are no contribution guidelines in the project yet.

**TODO:** Add a `CONTRIBUTING.md` file with guidelines for contributing to the project.
