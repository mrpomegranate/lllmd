# Imports
import sys
import tiktoken
import datetime
import json
import os

# Import classes
from modules.data.data_handler import DataHandler
from modules.interface.summary_llm import Summarizer
from modules.interface.agent_rag import MedicalResearchAgent

# Helper function
def group_summaries_into_chunks(summaries, max_tokens=3000):
    """
    Takes a list of summary strings and groups them into
    larger "super-chunks" that are under the token limit.
    """
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
    except:
        encoder = tiktoken.get_encoding("gpt2")

    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    separator = "\n\n--- (End of Section) ---\n\n"
    separator_tokens = len(encoder.encode(separator))

    # Iterate through the summary
    for summary in summaries:
        summary_tokens = len(encoder.encode(summary))

        # Check if this summary is too big
        if summary_tokens > max_tokens:
            print(f"Warning: A single summary ({summary_tokens} tokens) is too large. Truncating.")
            chunks.append(encoder.decode(encoder.encode(summary)[:max_tokens]))
            continue

        tokens_to_add = summary_tokens

        # Check if chunk isn't empty
        if current_chunk:
            # Put separator
            tokens_to_add += separator_tokens

        if current_chunk_tokens + tokens_to_add > max_tokens:
            # Bank the current chunk
            chunks.append(separator.join(current_chunk))
            # Start a new chunk
            current_chunk = [summary]
            current_chunk_tokens = summary_tokens
        else:
            # Add to the current chunk
            current_chunk.append(summary)
            current_chunk_tokens += tokens_to_add

    # Bank the last chunk
    if current_chunk:
        chunks.append(separator.join(current_chunk))

    print(f"Grouped {len(summaries)} summaries into {len(chunks)} 'super-chunks'.")
    return chunks


def clean_json_output(text):
    """
    Cleans the agent output to ensure it is valid JSON.
    Removes markdown formatting and finds the actual JSON brackets.
    """
    # Remove markdown code blocks if present
    text = text.replace("```json", "").replace("```", "")

    # Find the start ({) and end (}) of the JSON object
    # This handles cases where the agent says "Here is your JSON: {...}"
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index != -1 and end_index != -1:
        text = text[start_index: end_index + 1]

    return text.strip()

# Metadata
pipeline_metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "models": {
            "summarizer_1": "Mistral-7B-Instruct-v0.2",
            "summarizer_2": "Mistral-7B-Instruct-v0.2",
            "synthesizer": "Mistral-7B-Instruct-v0.2"
        },
        "processing_stats": {
            "case_id": 0,
            "total_initial_chunks": 0,
            "reduce_levels": 0
        }
    }

# Step 1: Initializing Data Handler
print("--- Initializing Data Handler ---")

# Initialize DataHandler
data_handler = DataHandler(data_dir="../storage", chunk_size_tokens=7000)  # 3000 7000

# Step 2: Get a note
# Getting notes from patient and divided them in chunks
case_id, note_chunks = data_handler.get_chunks_by_case(0)

# Check if it got the notes
if case_id is None:
    # Exit if no notes where gotten
    print(f"Error: {note_chunks}")
    sys.exit(1)

# Step 3: Initialize all models
# We do this once at the start to save time
print("\n--- Pre-loading all models... ---")

# TEST 1: BioMistral + Llama-Med42
# Initialize BioMistral
# First LLM summarizer
# print("Loading BioMistral-7B...")
# summarizer_1 = Summarizer(
#     repo_id="BioMistral/BioMistral-7B-GGUF",
#     gguf_filename="ggml-model-Q4_K_M.gguf"
# )

# Initialize Llama3-Med
# Second LLM summarizer
# print("Loading Llama3-Med42-8B...")
# summarizer_2 = Summarizer(
#     repo_id="SandLogicTechnologies/Llama3-Med42-8B-GGUF",
#     gguf_filename="Llama3-Med42-8B-Q4_K_M.gguf"
# )

# TEST 2: Meditron + OpenBioLLM
# print("Loading Meditron-7B...")
# summarizer_1 = Summarizer(
#     repo_id="TheBloke/meditron-7B-GGUF",
#     gguf_filename="meditron-7b.Q4_K_M.gguf",
#     chat_format="llama-2"
# )
#
# print("Loading OpenBioLLM-Llama3-8B...")
# summarizer_2 = Summarizer(
#     repo_id="aaditya/OpenBioLLM-Llama3-8B-GGUF",
#     gguf_filename="openbiollm-llama3-8b.Q4_K_M.gguf",
#     chat_format="llama-3"
# )

# TEST 3: BioMistral + Meditron
# print("Loading BioMistral-7B...")
# summarizer_1 = Summarizer(
#     repo_id="BioMistral/BioMistral-7B-GGUF",
#     gguf_filename="ggml-model-Q4_K_M.gguf"
# )
#
# print("Loading Meditron-7B...")
# summarizer_2 = Summarizer(
#     repo_id="TheBloke/meditron-7B-GGUF",
#     gguf_filename="meditron-7b.Q4_K_M.gguf"
# )

# TEST 4: General LLM + General LLM
summarizer_1 = Summarizer(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    gguf_filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

summarizer_2 = Summarizer(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    gguf_filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

# Initialize Mistral
# LLM to summarize the summaries from both LLMs
synthesizer = Summarizer(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    gguf_filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

print("All models loaded.")

# Step 4: MAP Step - Summarize each chunk
print(f"\n--- [Step 4: MAP - Summarizing {len(note_chunks)} chunks] ---")

# Store our final synthesized chunk summaries here
final_chunk_summaries = []

# Record the chunks count
pipeline_metadata["processing_stats"]["total_initial_chunks"] = len(note_chunks)

# Iterate through the chunks
for i, chunk in enumerate(note_chunks):
    print(f"\n--- Processing Chunk {i + 1}/{len(note_chunks)} ---")

    # Use medical LLMs to summarise patients notes current chunk
    SUMMARIZER_SYSTEM_PROMPT = ("You are a specialized medical assistant. "
                                "Your task is to provide a concise, clinically "
                                "accurate summary of the following patient note. "
                                "Focus on key diagnoses, medications, and findings.")

    SUMMARIZER_USER_PROMPT = f"Please summarize this note:\n\n[NOTE]\n{chunk}\n[/NOTE]"

    chunk_summary_1 = summarizer_1.generate_response(SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_USER_PROMPT)
    chunk_summary_2 = summarizer_2.generate_response(SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_USER_PROMPT)

    # Get both summaries and use an LLM to give a final summary of the current chunk
    SYNTHESIZER_SYSTEM_PROMPT = ("You are a helpful text-editing assistant. "
                                 "Your job is to combine two summaries of "
                                 "the same note into a single, de-duplicated, "
                                 "and coherent final summary. Do not add any new information.")

    SYNTHESIZER_USER_PROMPT = f"[SUMMARY 1]\n{chunk_summary_1}\n[/SUMMARY 1]\n\n[SUMMARY 2]\n{chunk_summary_2}\n[/SUMMARY 2]\n\nUnified Summary:"

    final_chunk_summary = synthesizer.generate_response(SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_PROMPT, max_tokens=350)

    # Output final summary fo the chunk
    print(f"--- Synthesized Summary for Chunk {i + 1} ---")
    print(final_chunk_summary)

    # Store the final chunk summary
    final_chunk_summaries.append(final_chunk_summary)

# Step 5: Combine all chunk summaries
print(f"\n\n--- [Step 5: Combine all chunks] ---")

current_level_summaries = final_chunk_summaries
level = 1

# This loop runs as long as we have more than 1 summary
# Loop through all the summary
# Turn all chunk summaries into 1 final summary
while len(current_level_summaries) > 1:
    print(f"\n--- Processing Reduce Level {level} ---")
    print(f"   Input: {len(current_level_summaries)} summaries")

    # Track how deep the hierarchy goes
    pipeline_metadata["processing_stats"]["reduce_levels"] = level

    # Group the summaries into new super chunks
    super_chunks = group_summaries_into_chunks(current_level_summaries, max_tokens=3000)

    next_level_summaries = []

    # Run the Reduce-Map
    REDUCE_SYSTEM_PROMPT = ("You are a senior medical editor. Your job is to "
                            "take a collection of summaries from a patient's "
                            "case and synthesize them into one single, "
                            "comprehensive summary. Focus on the complete "
                            "patient picture, key diagnoses, and treatment "
                            "progression. Be concise and clear.")

    # Iterate through the super chunks
    for i, chunk in enumerate(super_chunks):
        print(
            f"   ... Processing super-chunk {i + 1}/{len(super_chunks)}")
        REDUCE_USER_PROMPT = f"Please create a single, unified summary from the following summary sections:\n\n{chunk}\n\nUnified Summary:"

        # Reuse main synthesizer model
        new_summary = synthesizer.generate_response(REDUCE_SYSTEM_PROMPT, REDUCE_USER_PROMPT, max_tokens=1000)
        next_level_summaries.append(new_summary)

    # Set up for the next pass
    current_level_summaries = next_level_summaries
    level += 1

# Step 6: Final brief
final_brief = current_level_summaries[0]

# Output the final summary
print("\n" + "=" * 20)
print("--- [Step 6: Final Actionable Brief] ---")
print("=" * 20)
print(final_brief)

# Step 7: Agentic RAG

# Tavily API key
TAVILY_KEY = "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"

# Check if there's a tavily key
if "tvly-" not in TAVILY_KEY:
    print("\nNo API Key")
    final_json = None
else:
    print("\n\n--- [Step 7: STAGE 3 - Agentic RAG] ---")
    print("Initializing Agent connected to Local LLM Server...")

    try:
        agent_system = MedicalResearchAgent(tavily_api_key=TAVILY_KEY)
        # Pass the final brief from Step 6 into the agent
        verified_report = agent_system.research_and_verify(final_brief)

        clean_report = clean_json_output(verified_report)

        try:
            agent_json_data = json.loads(clean_report)
        except json.JSONDecodeError:
            # Save the string so you don't lose data
            print("\nAgent output was not perfect JSON. Saving raw text to avoid crash.")
            agent_json_data = {"raw_output": verified_report}

        final_output = {
            "system_metadata": pipeline_metadata,
            "generated_brief": final_brief,
            "clinical_analysis": agent_json_data
        }

        # Get date and time
        now = datetime.datetime.now()
        time_str = now.strftime("%m-%d-%Y_%H-%M-%S")

        # Get models names
        m1 = pipeline_metadata["models"]["summarizer_1"].replace(".gguf","").replace("-GGUF", "")
        m2 = pipeline_metadata["models"]["summarizer_2"].replace(".gguf","").replace("-GGUF", "")

        # File name
        filename = f"{m1}_{m2}_case_{case_id}_{time_str}.json"

        # Create logs folder
        os.makedirs("logs", exist_ok=True)

        full_file_path = os.path.join("logs", filename)

        # Save JSON file
        with open(full_file_path, 'w') as f:
            json.dump(final_output, f, indent=4)

        print(f"\nResults saved to: {full_file_path}")
        print(json.dumps(final_output, indent=2))


        print("\n" + "=" * 30)
        print("--- VERIFIED EVIDENCE REPORT ---")
        print("=" * 30)
        print(verified_report)

    except Exception as e:
        print(f"Agent failed to run.")
        print(f"Error: {e}")
