from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileWriterTool
import os
from datetime import datetime
import time

load_dotenv()



# AGENT SECTION

@CrewBase
class AiNews():
	"""AiNews crew"""
	
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	groq_llm1 = LLM(
		model=os.getenv("MODEL"),
		temperature=0.7,
		max_tokens=500,
		api_key=os.getenv("GROQ_API_KEY1")
	)

	groq_llm2 = LLM(
		model="groq/llama-3.1-8b-instant",
		temperature=0.7,
		max_tokens=500,
		api_key=os.getenv("GROQ_API_KEY2")
	)

	gemini_llm = LLM(
		model=os.getenv("MODELGEMINI"),
		temperature=0.7,
		max_tokens=500,
		api_key=os.getenv("GEMINI_API_KEY")
	)


	#agents are defined below

	@agent
	def retrieve_news(self) -> Agent: # this agent search the web using SerperDevTool()
		return Agent(
			config=self.agents_config['retrieve_news'],
			tools = [SerperDevTool()],
			llm = self.gemini_llm,
			verbose=True
		)

	@agent
	def website_scrapper(self) -> Agent: # this agent scrapes the website fetched from retrieve_news agent
		return Agent(
			config=self.agents_config['website_scrapper'],
			tools = [ScrapeWebsiteTool()],
			llm = self.gemini_llm,
			verbose=True
		)

	@agent
	def ai_news_writer(self) -> Agent: # This agent is responsible for writing or summarizing the content extracted from the website_scrapper agent.
		return Agent(
			config=self.agents_config['ai_news_writer'],
			tools = [],
			llm=self.gemini_llm,
			verbose=True
		)

	@agent
	def file_writer(self) -> Agent: #this agent is responsible for writing file to disc.
		return Agent(
			config=self.agents_config['file_writer'],
			tools = [FileWriterTool()],
			llm=self.groq_llm2,
			verbose=True,
			output_file=f'news/{datetime.now().strftime("%Y-%M-%D")}_news_article.md'
		)

# TASKS SECTION

	@task
	def retrieve_news_task(self) -> Task:
		return Task(
			config=self.tasks_config['retrieve_news_task'],
		)

	@task
	def website_scrape_task(self) -> Task:
		return Task(
			config=self.tasks_config['website_scrape_task'],
		)


	@task
	def ai_news_write_task(self) -> Task:
		return Task(
			config=self.tasks_config['ai_news_write_task'],
		)
	

	@task
	def file_write_task(self) -> Task:
		import random
		num = random.randint(1, 100) # Format the date as YYYY-MM-DD
		return Task(
			config=self.tasks_config['ai_news_write_task'],
			output_file=f'news/{num}_news_article.md'
		)
	
	
	

	@crew
	def crew(self) -> Crew:
		"""Creates the AiNews crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)

