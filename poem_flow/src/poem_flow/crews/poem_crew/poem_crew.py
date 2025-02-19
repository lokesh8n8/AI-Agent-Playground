from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileWriterTool
import os
from datetime import datetime
import time

load_dotenv()


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class PoemCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

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
    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def poem_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["poem_writer"],
            llm = self.gemini_llm,
            verbose= True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def write_poem(self) -> Task:
        return Task(
            config=self.tasks_config["write_poem"],
            llm=self.groq_llm2,
            verbose=True
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
