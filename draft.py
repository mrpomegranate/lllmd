from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    pipeline
)
import torch
from tqdm import tqdm
import os
import pandas as pd

# Enable verbose logging
logging.set_verbosity_info()

# Enable hf_transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'
os.environ['HF_HUB_DISABLE_XET']='1'


# --- 1. Load Dataset ---

df = pd.read_csv('dataset\\nbme-score-clinical-patient-notes\patient_notes.csv')

# Assuming columns: case_num, note
# Combine notes per case_num if there are multiple
grouped = (
    df.groupby("case_num")["pn_history"]
    .apply(lambda x: " ".join(x.astype(str)))
    .reset_index()
)


# quantize for bio mistral
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# # bio mistral
# bio_mistral_clinical_7b = "ZiweiChen/BioMistral-Clinical-7B"
# tokenizer = AutoTokenizer.from_pretrained(bio_mistral_clinical_7b)
# bio_mistral_clinical_7b_model = AutoModelForCausalLM.from_pretrained("ZiweiChen/BioMistral-Clinical-7B",
#                                                                      quantization_config=bnb_config)


# Then update the tokenizer and model loading:
model_name = "BioMistral/BioMistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto',
    use_safetensors=True
)

# bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# # model_name = "Falconsai/medical_summarization"
# model_name = "google/flan-t5-large"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto"
# )




def chunk_text(text, tokenizer, max_tokens=512, prompt_tokens=100):
    """
    Split long text into smaller chunks safely based on tokenizer length.
    Reserve space for prompt tokens.
    """
    available_tokens = max_tokens - prompt_tokens
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [
        tokenizer.decode(tokens[i:i + available_tokens], skip_special_tokens=True)
        for i in range(0, len(tokens), available_tokens)
    ]
    return chunks


def summarize(text, model, tokenizer, max_new_tokens=256):
    """
    Summarize one clinical note using BioMistral-Clinical-7B.
    """

    summaries = []
    chunks = chunk_text(text, tokenizer, max_tokens=512, prompt_tokens=prompt_tokens)

    for chunk in chunks:

        prompt = (
            "You are a highly skilled medical summarization assistant.\n"
            "Your task is to read the following clinical note and generate a concise summary in few sentences.\n"
            "Focus on:\n"
            "- The patient's main complaint\n"
            "- Relevant medical history and family history\n"
            "- Associated symptoms\n"
            "- Any concerning findings\n\n"
            f"Clinical Note:\n{chunk}\n\n"
            "Summary:"
        )

        # prompt = (
        #     f"Summarize the following clinical note in 2-3 sentences. Focus on the patient's main complaint, relevant medical and family history, associated symptoms, and any concerning findings.\n\n"
        #     f"Clinical Note:\n{chunk}\n\n"
        #     "Summary:"
        # )
        prompt_tokens = len(tokenizer.encode(prompt, truncation=False))

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                num_beams=4,
                repetition_penalty=1.2,
                do_sample=True,
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary.split("Summary:")[-1].strip()
        summaries.append(summary)
        print('------------------------------------------------------------------------------------')
        print("Summarizing the following Chunk: ")
        print(chunk)
        print("Summarized Chunk: ")
        print(summary)

    combined_summary = " ".join(summaries)

    if len(chunks) > 1:
        # optional: refine multi-chunk summaries into one

        refine_prompt = (
            "You are a medical summarization assistant.\n"
            "Combine and refine the following partial summaries into one clear, concise clinical summary in 3–4 sentences:\n\n"
            f"{combined_summary}\n\nFinal Summary:"
        )

        refine_inputs = tokenizer(refine_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            refine_outputs = model.generate(
                **refine_inputs,
                max_new_tokens=250,
                temperature=0.3,
                num_beams=4,
            )
        refined_summary = tokenizer.decode(refine_outputs[0], skip_special_tokens=True)
        refined_summary = refined_summary.split("Final Summary:")[-1].strip()
        return refined_summary.strip()

    return combined_summary.strip()

# --- Summarize each pn_history ---
# summaries = []
# for i, note in tqdm(enumerate(df['pn_history']), total=len(df), desc="Summarizing pn_history"):
#     if pd.isna(note) or not isinstance(note, str) or len(note.strip()) == 0:
#         summaries.append("")
#         continue

summaries=[]
note = grouped['pn_history'][0]
print("Original text (first 400 chars):")
print(note[:400] + "...\n")

try:
    summary = summarize(note, model, tokenizer)
except Exception as e:
    summary = f"[Error: {str(e)}]"

summaries.append(summary)


    # --- Add summaries to DataFrame and save ---
df['googleT5Large_summary'] = summaries

output_path = "summarized_pn_history_biomistral.csv"
df.to_csv(output_path, index=False)
print(f"Summarization complete! Saved to {output_path}")

# BioMistral-7B
# Llama3-Med42-8B