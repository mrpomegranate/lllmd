import sys

from modules.data.data_handler import DataHandler
from modules.interface.summary_llm import Summarizer

print("--- Initializing Data Handler ---")
data_handler = DataHandler(data_dir="storage")  # Get .csv file

# Get note test
note_id, note_text = data_handler.get_note_by_id(16)
print(f"\n--- Loaded Note ID: {note_id} ---")
print(note_text)
print("-" * 30)

# Summarization

# Model 1: BioMistral-7B
# We use Q4 4-bit quant file size is ~4.1 GB
print("\n--- Loading Summarizer 1: BioMistral ---")
summarizer_1 = Summarizer(repo_id="BioMistral/BioMistral-7B-GGUF", gguf_filename="ggml-model-Q4_K_M.gguf")
summary_1 = summarizer_1.summarize(note_text)
print("\n--- SUMMARY 1 (BioMistral) ---")
print(summary_1)

# Model 2: Llama-3-Med42-8B
# We use 8B model it's a ~4.6 GB for a Q4
print("\n--- Loading Summarizer 2: Llama-3-Med42 ---")
summarizer_2 = Summarizer(repo_id="SandLogicTechnologies/Llama3-Med42-8B-GGUF", gguf_filename="Llama3-Med42-8B-Q4_K_M.gguf")

summary_2 = summarizer_2.summarize(note_text)
print("\n--- SUMMARY 2 (Llama-3-Med42) ---")
print(summary_2)
print("\n--- Initializing Synthesizer (Mistral-Instruct) ---")


# Summarize both medical summaries
synthesizer = Summarizer(repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", gguf_filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf")

print("\n" + "="*20)
print("  STAGE 2: SYNTHESIS  ")
print("="*20)
try:

    # Define synthesis prompt
    SYNTHESIZER_SYSTEM_PROMPT = """
    You are a text-editing assistant. Your job is to combine two
    summaries of the same note into a single, de-duplicated, and
    coherent final summary. Do not add any new information.
    """.strip()

    SYNTHESIZER_USER_PROMPT = f"""
    Please merge the following two summaries into one unified summary.
    
    [SUMMARY 1]
    {summary_1}
    [/SUMMARY 1]
    
    [SUMMARY 2]
    {summary_2}
    [/SUMMARY 2]
    
    Unified Summary:
    """

    # Run synthesis
    final_summary = synthesizer.generate_response(SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_PROMPT, max_tokens=350)

    # Final summary
    print("\n" + "=" * 20)
    print("  FINAL SYNTHESIZED SUMMARY  ")
    print("=" * 20)
    print(final_summary)


except Exception as e:
    print(f"\n--- An error occurred during synthesis ---")
    print(f"Error details: {e}")
    print("This might be a 404 error for the synthesizer model. Check the filename.")
    sys.exit(1)
