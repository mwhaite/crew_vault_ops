# Usage

This document explains how to use the `VaultOpsTool` with `crewai`.

## 1. Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Create a Crew

Next, create a Python script to define your `crewai` crew. Here is an example:

```python
from crewai import Agent, Task, Crew
from vault_ops.vault_ops_tool import VaultOpsTool

# Instantiate the tool
vault_tool = VaultOpsTool()

# Define the agent
researcher = Agent(
  role='Obsidian Vault Researcher',
  goal='Use the vault to answer questions',
  backstory='You are an expert in navigating and understanding the content of an Obsidian vault.',
  tools=[vault_tool],
  verbose=True,
  allow_delegation=False
)

# Define the task
task = Task(
  description='What is CrewAI?',
  agent=researcher
)

# Create and run the crew
crew = Crew(
  agents=[researcher],
  tasks=[task],
  verbose=2
)

result = crew.kickoff()

print("************************")
print("** Here is the result")
print("************************")
print(result)
```

## 3. Run the Crew

Finally, run the script:

```bash
python sample_crew.py
```

This will run the crew and print the result to the console.
