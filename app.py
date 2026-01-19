import os
import gc
import re
import shutil
import pandas as pd
import torch
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ML & Agent Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# ==============================================================================
# 1. CORE LOGIC CLASSES
# ==============================================================================

class ClinicalDataProcessor:
    def __init__(self):
        self.df = None
        self.file_path = "patient_notes.csv"

    def load_data(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError("File not found.")
        self.df = pd.read_csv(path)
        print(f"✅ Data Loaded. Rows: {len(self.df)}")

    def get_cases(self) -> List[int]:
        if self.df is None: return []
        return [int(x) for x in sorted(self.df['case_num'].unique().tolist())]

    def get_case_preview(self, case_num: int, limit: int = 5) -> List[Dict[str, Any]]:
        if self.df is None: return []
        subset = self.df[self.df['case_num'] == case_num]
        if limit > 0:
            subset = subset.head(limit)
        return subset.where(pd.notnull(subset), None).to_dict(orient='records')

    def get_combined_notes(self, case_num: int, limit: Optional[int] = None) -> str:
        if self.df is None: return ""
        subset = self.df[self.df['case_num'] == case_num]
        if limit:
            subset = subset.head(limit)
        return "\n".join(subset['pn_history'].astype(str).tolist())

    def smart_chunking(self, text: str, chunk_size: int = 2048, overlap_percentage: float = 0.15) -> List[str]:
        if not text: return []
        overlap = int(chunk_size * overlap_percentage)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        return splitter.split_text(text)

# ==========================================
# BLOCK 2: MODULAR LOCAL LLM SUMMARIZER
# ==========================================
class ClinicalSummarizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available on: {self.device}")

    def _load_model(self, model_id: str):
        print(f"--- Loading {model_id} (8-bit) ---")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        hf_token = os.getenv('HF_TOKEN')

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

            # CRITICAL FIX: Set pad_token before pipeline creation
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise

    def _cleanup(self, model, tokenizer):
        """
        Forcefully clears VRAM to allow the next model to load.
        """
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("--- VRAM Cleared ---")

    def summarize_chunks(self, chunks: List[str], model_id: str) -> str:
            """
            Iteratively updates a single summary with new information from each chunk.
            This prevents repetition by letting the model 'see' what it already wrote.
            """
            if not chunks: return "No text to summarize."

            if "BioMistral" in model_id:
                try:
                    model, tokenizer = self._load_model(model_id)

                    # PIPELINE: Ensure strict decoding to prevent "wandering"
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=600,
                        do_sample=True,
                        temperature=0.2,     # LOWER temp to make it strictly follow instructions
                        top_p=0.9,
                        repetition_penalty=1.1,
                        return_full_text=False
                    )

                    current_summary = "No prior information."

                    for i, chunk in enumerate(chunks):
                        print(f"Processing chunk {i+1}/{len(chunks)} with {model_id}...")

                        if i == 0:
                            # FIX 1: Simplified, punchy instruction
                            instruction = (
                                "You are a Clinical Documentation Specialist. "
                                "Extract a structured clinical abstract from the note below. "
                                "Use EXACTLY these headers: Demographics, Medical History, Family History, Social History, HPI."
                            )
                            input_text = f"CLINICAL NOTE:\n{chunk}"

                            # FIX 2: PREFILL THE PROMPT
                            # We end the prompt with "Demographics:" to force the model into the format immediately.
                            prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n\nDemographics:"
                        else:
                            instruction = "Update the existing summary with new info. Keep the exact same format."
                            input_text = f"EXISTING SUMMARY:\n{current_summary}\n\nNEW NOTE:\n{chunk}"
                            prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n\nUpdated Summary:"

                        try:
                            outputs = pipe(prompt)
                            new_text = outputs[0]['generated_text'].strip()

                            # FIX 3: Re-attach the prefilled header if it was the first chunk
                            if i == 0:
                                new_text = "Demographics: " + new_text

                            print(f"--- Chunk {i+1} Output ({len(new_text)} chars) ---")

                            # Safety check to ensure we don't overwrite with empty garbage
                            if len(new_text) > 20:
                                current_summary = new_text
                            else:
                                print("⚠️ Warning: Model returned empty text. Skipping update.")

                        except Exception as e:
                            print(f"Error generating chunk {i}: {e}")

                    self._cleanup(model, tokenizer)
                    return current_summary

                except Exception as e:
                    print(f"Summarizer Critical Error: {e}")
                    raise e
            else:
                try:
                    model, tokenizer = self._load_model(model_id)

                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.1,
                        repetition_penalty=1.2,
                        return_full_text=False
                    )

                    # --- STATE: This holds the evolving summary ---
                    current_summary = "No prior information."

                    for i, chunk in enumerate(chunks):
                        print(f"Processing chunk {i+1}/{len(chunks)} with {model_id} (Refine Step)...")

                        # --- DYNAMIC PROMPT: Changes based on whether we have a summary yet ---
                        if i == 0:
                            # First chunk: Standard Extraction
                            system_instruction = """You are an expert Clinical Documentation Specialist.

                            GOAL: Extract key clinical data into a structured abstract.
                            STRICT FORMAT:
                            Demographics: [Age, Sex]
                            Medical History: [Conditions, Meds, Allergies]
                            Family History: [Conditions]
                            Social History: [Lifestyle]
                            Gynocological History: [If Female Patient or NA],
                            History of Present Illness: [Symptoms, Timeline]"""

                            user_message = f"PATIENT NOTE:\n{chunk}\n\nSUMMARY:"
                        else:
                            # Subsequent chunks: UPDATE the existing summary
                            system_instruction = """You are an expert Clinical Documentation Specialist.
                            GOAL: You are given an EXISTING SUMMARY and a NEW NOTE from a different doctor.
                            INSTRUCTION: Update the Existing Summary with ONLY NEW information found in the New Note.
                            - If the New Note repeats information already in the Summary, IGNORE IT.
                            - Keep the same strict format.
                            - Merge findings logically."""

                            user_message = f"EXISTING SUMMARY:\n{current_summary}\n\nNEW NOTE:\n{chunk}\n\nUPDATED SUMMARY:"

                        # Construct Prompt
                        try:
                            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_message}]
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        except:
                            prompt = f"[INST] {system_instruction}\n\n{user_message} [/INST]"

                        # Generate
                        output = pipe(prompt)[0]['generated_text'].strip()

                        # Update the state variable with the new, refined summary
                        if output:
                            current_summary = output

                    self._cleanup(model, tokenizer)
                    return current_summary

                except Exception as e:
                    if "CUDA out of memory" in str(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        raise HTTPException(status_code=503, detail="GPU Out of Memory.")
                    raise e

    def meta_summarize(self, summary_a: str, summary_b: str, aggregator_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3") -> str:
        """
        Takes the two summaries and fuses them.
        """
        print("--- Generating Meta-Summary ---")
        combined_input = f"Summary from Model A:\n{summary_a}\n\nSummary from Model B:\n{summary_b}"

        model, tokenizer = self._load_model(aggregator_model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000, return_full_text=False)

        prompt = f"""You are a Clinical Data API capable of merging medical summaries into structured JSON.

        INSTRUCTION:
        Synthesize the provided summaries into a single cohesive JSON object.

        STRICT OUTPUT SCHEMA:
        You must return a VALID JSON object with exactly these keys. Do not add keys. Do not add markdown formatting.

        {{
            "Demographics": "String containing Age and Sex",
            "History of Present Illness": ["List", "of", "key", "points"],
            "Medical History": ["List", "of", "history", "meds", "allergies"],
            "Gynocological History": ["if", "the", "pateint", "is", "female"],
            "Family History": "String summary of hereditary conditions",
            "Social History": "String summary of lifestyle/substance use",
            "Physical Exam": "String summary of vitals and findings (or 'Not provided')",
            "Diagnostics": "Previous diagnostics, labs, x-rays and other tests if available"
        }}

        RULES:
        1. Output must be parsable JSON.
        2. Escape all special characters inside strings.
        3. Do NOT include an Assessment or Plan.
        4. Do NOT include markdown ticks (```json). Just the raw JSON object.

        INPUT DATA:
        {combined_input}

        JSON OUTPUT:
        """

        output = pipe(prompt)[0]['generated_text']

        # --- FAIL-SAFE CLEANING LOGIC ---
        # If the prompt is still in the output, split by the last line of your prompt
        split_marker = 'JSON OUTPUT:'

        if split_marker in output:
            final_text = output.split(split_marker)[-1].strip()
        else:
            final_text = output.strip()

        self._cleanup(model, tokenizer)
        return final_text


class MedicalSearchTool(BaseTool):
    name: str = "Medical Search Tool"
    description: str = "Useful for searching the web for clinical guidelines, medical definitions, and differential diagnoses."

    def _run(self, query: str) -> str:
            try:
                # Attempt the search
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=4))

                    # FIX 1: Handle empty results explicitly
                    if not results:
                        return "Search returned no results. Please rely on your internal medical knowledge for this section."

                    # FIX 2: Better formatting
                    formatted_results = []
                    for r in results:
                        formatted_results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nSource: {r.get('href')}")

                    return "\n\n".join(formatted_results)

            except Exception as e:
                # FIX 3: Catch library errors (like rate limits) so the Agent doesn't crash
                print(f"Search Error: {e}")
                return "Search tool is currently unavailable. Please proceed using your internal medical expertise and cite 'Internal Knowledge'."

class DiagnosisCrew:
    def __init__(self):
        self.search_tool = MedicalSearchTool()
        self.llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ],
            temperature=0.1
        )

    def run_diagnosis_search(self, clinical_summary: str):

        # --- Agent 1: The Diagnostician ---
        medical_researcher = Agent(
        role='Senior Medical Diagnostician',
        goal=(
            "You are a specialist medical researcher. "
            "Your task is NOT to provide definitive diagnoses, but to generate "
            "patient-specific differential diagnoses with risk stratification, "
            "clinical rationale, and recommended next steps based on trusted sources."
        ),
        backstory=(
            "You consult high-quality medical databases such as PubMed, "
            "NICE guidelines, Mayo Clinic, MedlinePlus. "
            "You integrate patient demographics, symptoms, lifestyle, medications, "
            "and family history into your reasoning."
        ),
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool],
            llm=self.llm,
        )

        # --- Agent 2: The Critic ---
        medical_critic = Agent(
            role='Expert Medical Review Board Critic',
            goal='Critique the differential diagnosis for bias, likelihood, and missing perspectives. ' \
            'Assign a 0-10 confidence score.',
            backstory="""You are a world-renowned multi-disciplinary medical consultant (Dr. House archetype).
            You review cases to ensure no rare disease is missed and that
            common diseases aren't assumed without evidence.
            You act as the quality control layer.
            """,
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool], # Critic can also search to verify claims
            llm=self.llm,

        )

        # --- Task 1: Generate Diagnosis ---
        research_task = Task(
            description=f"""
                Analyze the following Patient Case Summary:
                {clinical_summary}

                1. Extract key risk factors: age, medications, lifestyle, family history, comorbidities.
                2. Generate a **prioritized differential diagnosis**, explaining why each is likely.
                3. Identify **red flags** that need urgent attention.
                4. Suggest **next steps**, including:
                - Recommended labs, imaging, or monitoring
                - Lifestyle or medication modifications
                - Specialist referrals if needed
                5. Include evidence from credible sources (PubMed, Mayo Clinic, MedlinePlus, NICE guidelines)
                with citations or links.

                Return a structured report with the following sections:
                - Patient Summary
                - Risk Factors
                - Red Flags
                - Differential Diagnosis (prioritized)
                - Supporting Evidence
                - Recommended Next Steps

                Use the following websites
                https://www.mayoclinic.org/
                https://medlineplus.gov/
                https://pubmed.ncbi.nlm.nih.gov/
                https://familydoctor.org/
                https://www.nice.org.uk/
            """,
            expected_output=("A comprehensive Markdown medical report containing Diagnosis, Evidence, and Recommended Next Steps."),
            agent=medical_researcher,
            verbose=True
        )


        # --- Task 2: Critique & Format (JSON Enforced) ---
        critique_task = Task(
            description=f"""
            Review the Draft Diagnosis Report provided by the {medical_researcher} agent.

            YOUR TASKS:
            1. Evaluate each proposed diagnosis.
            2. Assign a 'Critic Score' (0-10) based on how well the evidence fits.
            3. Provide a 'Critic Explanation' for your score.
            4. REVIEW the 'Recommended Next Steps'. Assign a score (0-10) to each step's relevance and explain why.
            5. Only use reputable medical journals, websites and resources.

            STRICT OUTPUT: Return ONLY a valid JSON object with this exact structure:
            {{
                "patient_summary": "One sentence summary of the patient context",
                "differential_diagnoses": [
                    {{
                        "name": "Diagnosis Name",
                        "likelihood": "High/Medium/Low",
                        "rationale": "Medical reasoning...",
                        "critic_score": 8,
                        "critic_explanation": "Why this score..."
                    }}
                ],
                "red_flags": ["Flag 1", "Flag 2"],
                "risk_factors": ["Factor 1", "Factor 2"],
                "recommended_next_steps": [
                    {{
                        "name": "Step Name (e.g., MRI)",
                        "critic_score": 9,
                        "critic_explanation": "Why this is critical"
                    }}
                ]
            }}
            """,
            expected_output="A valid json object",
            agent=medical_critic,
            verbose=True
        )
        crew = Crew(
            agents=[medical_researcher, medical_critic],
            tasks=[research_task, critique_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        return result

# ==============================================================================
# 2. GLOBAL STATE & LIFECYCLE
# ==============================================================================

processor = ClinicalDataProcessor()
summarizer = ClinicalSummarizer()
diagnosis_crew = DiagnosisCrew()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server Starting...")
    if os.path.exists("patient_notes.csv"):
        try: processor.load_data("patient_notes.csv")
        except: pass
    yield
    print("🛑 Server Stopping...")
    gc.collect()
    torch.cuda.empty_cache()

# ==============================================================================
# 2. APP SETUP
# ==============================================================================

processor = ClinicalDataProcessor()
summarizer = ClinicalSummarizer()
diagnosis_crew = DiagnosisCrew()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server Starting...")
    if os.path.exists("patient_notes.csv"):
        try: processor.load_data("patient_notes.csv")
        except Exception as e: print(f"⚠️ Could not auto-load data: {e}")
    yield
    print("🛑 Server Stopping...")
    gc.collect()
    torch.cuda.empty_cache()

app = FastAPI(title="MediMind API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- MODELS ---
class SummaryRequest(BaseModel):
    case_num: int
    patient_limit: Optional[int] = 5

class DiagnosisRequest(BaseModel):
    clinical_summary: str

# ==============================================================================
# 3. ENDPOINTS
# ==============================================================================

# --- NEW: Serve the UI directly ---
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the index.html file directly on localhost:8000"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found.</h1><p>Please ensure index.html is in the same folder as app.py</p>"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = "patient_notes.csv"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        processor.load_data(file_location)
        return {"message": "File uploaded successfully", "rows": len(processor.df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cases")
def get_cases():
    if processor.df is None: raise HTTPException(status_code=404, detail="No data loaded.")
    return {"cases": processor.get_cases()}

@app.get("/cases/{case_num}")
def preview_case(case_num: int, limit: int = 5):
    data = processor.get_case_preview(case_num, limit)
    if not data: raise HTTPException(status_code=404, detail="Case not found")
    return {"data": data}

@app.post("/summarize")
def run_summarization(req: SummaryRequest):
    try:
        raw_text = processor.get_combined_notes(req.case_num, req.patient_limit)
        if not raw_text: raise HTTPException(status_code=404, detail="No notes found.")
        chunks = processor.smart_chunking(raw_text)

        summary_a = summarizer.summarize_chunks(chunks, "BioMistral/BioMistral-7B")
        summary_b = summarizer.summarize_chunks(chunks, "m42-health/Llama3-Med42-8B")
        summary_final = summarizer.meta_summarize(summary_a, summary_b, aggregator_model_id='mistralai/Mistral-7B-Instruct-v0.3')

        return {"summary_a": summary_a, "summary_b": summary_b, "summary_final": summary_final}
    except Exception as e:
        print(f"Summarization Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diagnose")
def run_diagnosis(req: DiagnosisRequest):
    try:
        result = diagnosis_crew.run_diagnosis_search(req.clinical_summary)
        return {"diagnosis_report": str(result)}
    except Exception as e:
        print(f"Diagnosis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))