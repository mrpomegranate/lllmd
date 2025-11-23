import pandas as pd
from pathlib import Path
import tiktoken


class CaseDataHandler:
    """
    A class to load all notes for a single patient
    and provide them in token-limit-aware chunks.
    """

    def __init__(self, data_dir="../storage", chunk_size_tokens=3500):
        """
        Initializes the data handler by loading the dataset.
        :param data_dir: Folder where patient_notes.csv is.
        :param chunk_size_tokens: Target max token size for each chunk.
        """
        script_dir = Path(__file__).parent
        self.csv_path = script_dir.parent / data_dir / "patient_notes.csv"
        self.df = None
        self.chunk_size_tokens = chunk_size_tokens

        try:
            # Initialize token encoder
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Environments where tiktoken might have issues
            self.encoder = tiktoken.get_encoding("gpt2")

        self.load_data()

    def load_data(self):
        """
        Loads the patient_notes.csv file.
        :return:
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df = self.df[['pn_num', 'case_num', 'pn_history']]
            print(f"CaseDataHandler loaded {len(self.df)} total note entries.")
        except FileNotFoundError:
            print("=" * 50)
            print(f"Error: Could not find patient_notes.csv at {self.csv_path}")
            print(f" Please make sure your data is in a '{self.csv_path.parent}' folder.")
            print("=" * 50)
            self.df = pd.DataFrame(columns=['pn_num', 'case_num', 'pn_history'])

    def count_tokens(self, text):
        """Helper function to count tokens in a string."""
        return len(self.encoder.encode(text))

    def get_chunks_by_case(self, case_num):
        """
        Retrieves all notes for a case and splits them into
        chunks that respect the token limit.

        This method chunks intelligently on note boundaries.
        """
        # Check if the dataframe is empty
        if self.df.empty:
            return None, "Error: DataFrame is empty. Could not load data."

        # Get all notes for the case
        case_notes_df = self.df[self.df['case_num'] == case_num]
        if case_notes_df.empty:
            return None, f"Error: No notes found for case_num {case_num}."

        # Get a simple list of the note text
        note_list = case_notes_df['pn_history'].tolist()

        chunks = []
        current_chunk_notes = []  # List to store notes for the current chunk
        current_chunk_tokens = 0
        separator = "\n\n--- (New Note Entry) ---\n\n"
        separator_tokens = self.count_tokens(separator)

        # Iterate through each note and add it to a chunk
        for note in note_list:
            note_tokens = self.count_tokens(note)

            # Check if a single note is bigger than the limit
            if note_tokens > self.chunk_size_tokens:
                if current_chunk_notes:
                    # Truncate it and add it as its own chunk
                    chunks.append(separator.join(current_chunk_notes))
                    current_chunk_notes = []
                    current_chunk_tokens = 0

                print(f"Warning: A single note ({note_tokens} tokens) is larger than the {self.chunk_size_tokens} limit. Truncating.")
                encoded_note = self.encoder.encode(note)
                chunks.append(self.encoder.decode(encoded_note[:self.chunk_size_tokens]))
                continue

            # Check if adding this note and separator would pass the limit
            tokens_to_add = note_tokens
            if current_chunk_notes:
                tokens_to_add += separator_tokens

            if current_chunk_tokens + tokens_to_add > self.chunk_size_tokens:
                # Store the current chunk
                chunks.append(separator.join(current_chunk_notes))
                # Start a new chunk with the current note
                current_chunk_notes = [note]
                current_chunk_tokens = note_tokens
            else:
                # Add this note to the current chunk
                current_chunk_notes.append(note)
                current_chunk_tokens += tokens_to_add

        # Store the final chunk
        if current_chunk_notes:
            chunks.append(separator.join(current_chunk_notes))

        print(f"Case {case_num} split into {len(chunks)} chunk(s).")
        return case_num, chunks

    def __len__(self):
        if self.df is None: return 0
        return len(self.df)