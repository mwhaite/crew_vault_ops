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

print("\n\n########################")
print("## Here is the result")
print("########################\n")
print(result)
