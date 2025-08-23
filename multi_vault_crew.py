from crewai import Agent, Task, Crew
from vault_ops.vault_ops_tool import VaultOpsTool

# Create two instances of the VaultOpsTool, each with a different vault
vault_1_tool = VaultOpsTool(vault_path="./vault_1")
vault_2_tool = VaultOpsTool(vault_path="./vault_2")

# Define two agents, each with a different vault tool
vault_1_researcher = Agent(
  role='Vault 1 Researcher',
  goal='Use vault 1 to answer questions',
  backstory='You are an expert in navigating and understanding the content of vault 1.',
  tools=[vault_1_tool],
  verbose=True,
  allow_delegation=False
)

vault_2_researcher = Agent(
  role='Vault 2 Researcher',
  goal='Use vault 2 to answer questions',
  backstory='You are an expert in navigating and understanding the content of vault 2.',
  tools=[vault_2_tool],
  verbose=True,
  allow_delegation=False
)

# Define two tasks, one for each agent
task_1 = Task(
  description='What is in this vault?',
  agent=vault_1_researcher
)

task_2 = Task(
  description='What is in this vault?',
  agent=vault_2_researcher
)

# Create and run the crew
crew = Crew(
  agents=[vault_1_researcher, vault_2_researcher],
  tasks=[task_1, task_2],
  verbose=2
)

result = crew.kickoff()

print("\n\n########################")
print("## Here is the result")
print("########################\n")
print(result)
