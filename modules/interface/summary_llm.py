import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class Summarizer:
    """
    A class to generate summaries using a GGUF model via llama-cpp-python.
    This is optimized for Apple Silicon (M1/M2/M3) with Metal.
    """

    def __init__(self, repo_id, gguf_filename, chat_format=None, local_model_dir="storage/model_download"):
        """
        Initializes the summarizer by downloading and loading the GGUF model.
        :param repo_id: Hugging Face repo ID
        :param gguf_filename: The specific .gguf file to download
        :param local_model_dir: Directory to cache downloaded models
        """
        self.repo_id = repo_id
        self.gguf_filename = gguf_filename
        self.model_path = self._download_model(local_model_dir)

        print(f"Loading model from: {self.model_path}")

        # Load model into memory
        self.llm = Llama(model_path=self.model_path, n_gpu_layers=-1, n_ctx=9000, chat_format=chat_format, verbose=False)  # 4096 8194

        print(f"Model {gguf_filename} loaded successfully.")

    def _download_model(self, local_model_dir):
        """
        Downloads the GGUF model file from Hugging Face if it doesn't exist locally.
        :param local_model_dir: Directory to cache downloaded models
        """
        os.makedirs(local_model_dir, exist_ok=True)
        local_path = os.path.join(local_model_dir, self.gguf_filename)

        # Check if model has been downloaded
        if not os.path.exists(local_path):
            # Download model
            print(f"Model file not found. Downloading {self.gguf_filename}...")
            hf_hub_download(repo_id=self.repo_id, filename=self.gguf_filename, local_dir=local_model_dir)
            print("Download complete.")

        return local_path

    def generate_response(self, system_prompt, user_prompt, max_tokens=250):
        """
        A generic method to generate text based on any chat prompts.
        This is our new flexible method.
        :param system_prompt: The system prompt
        :param user_prompt: The user prompt
        :param max_tokens: The maximum number of tokens to generate
        """
        print(f"--- Generating response with {self.gguf_filename.split('.')[0]}... ---")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.3)

        content = response['choices'][0]['message']['content'].strip()
        return content

    def summarize(self, note_text, max_tokens=250):
        """
        A simple "shortcut" method for the specific task of
        medical note summarization.
        :param note_text: The note text to summarize
        :param max_tokens: The maximum number of tokens to generate
        """
        # Define specific prompts for task
        SUMMARIZER_SYSTEM_PROMPT = "You are a specialized medical assistant. Your task is to provide a concise, clinically accurate summary of the following patient note. Focus on key diagnoses, medications, and findings."
        SUMMARIZER_USER_PROMPT = f"Please summarize this note:\n\n[NOTE]\n{note_text}\n[/NOTE]"

        return self.generate_response(SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_USER_PROMPT, max_tokens=max_tokens)
