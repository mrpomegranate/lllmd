import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import TavilySearchTool


class MedicalResearchAgent:
    """
    Class to use crewai agent to verify clinical note summarization.
    """
    def __init__(self, tavily_api_key):
        """
        Initialise the agent with tavily_api_key.
        :param tavily_api_key:
        """
        os.environ["TAVILY_API_KEY"] = tavily_api_key  # API key
        self.llm = LLM(model="openai/mistral", base_url="http://localhost:8000/v1", api_key="sk-no-key-needed")  # Connect to local LLM server
        self.search_tool = TavilySearchTool()  # Initialize tools

    def research_and_verify(self, clinical_brief):
        """
        Takes the synthesized brief and runs an agentic search
        to verify facts and find guidelines.
        :param clinical_brief: Final summarization of clinical note
        """
        # Formatting the output
        json_structure = """
                {
                  "Patient Summary": {
                    "Demographics": "...",
                    "History of Present Illness": "...",
                    "Medical History": "...",
                    "Family History": "...",
                    "Social History": "...",
                    "Physical Exam": "...",
                    "Diagnostics": "..."
                  },
                  "Risk Factors": {
                    "Age": "...",
                    "Medications/Substances": "...",
                    "Lifestyle": "...",
                    "Family History": "...",
                    "Comorbidities": "..."
                  },
                  "Red Flags": [ "flag1", "flag2" ],
                  "Differential Diagnosis (prioritized)": [
                    {
                      "Diagnosis": "...",
                      "Likelihood": "High/Moderate/Low",
                      "Rationale": "...",
                      "Supporting Evidence": "URL"
                    }
                  ],
                  "Recommended Next Steps": {
                    "Immediate Actions": [],
                    "Diagnostics": [],
                    "Lifestyle Modifications": [],
                    "Medication Modifications": [],
                    "Specialist Referrals": [],
                    "Follow-up": []
                  }
                }
                """

        # Define the agent
        researcher = Agent(
            role='Senior Medical Research Specialist',
            goal='Verify clinical statements and find current treatment guidelines',
            backstory="""You are an expert medical researcher. Your job is to verify 
            clinical accuracy and find definitive treatment guidelines from reputable 
            sources (NIH, NICE, PubMed) for the identified conditions.""",
            verbose=True,
            memory=False,
            tools=[self.search_tool],
            llm=self.llm
        )

        # Define the task
        research_task = Task(
            description=f"""
            Analyze the following patient brief:

            "{clinical_brief}"

            1. Identify the 3 most critical clinical issues or diagnoses.
            2. For EACH issue, use the Search Tool to find the latest 
               guidelines (NIH, NICE, or similar).
            3. Verify if the treatment mentioned in the brief aligns with 
               standard guidelines.
               
            Use this exact JSON structure for your output:
            {json_structure}
            
            Do not add markdown formatting. Just return the raw JSON string.
            """,
            expected_output="A valid JSON object matching the provided schema.",
            agent=researcher
        )

        # Crew
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            process="sequential"
        )

        # Run the agent
        print("\n--- Agent Activated: Researching Guidelines ---")
        result = crew.kickoff()
        return result.raw