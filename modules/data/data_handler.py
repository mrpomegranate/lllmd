import pandas as pd
import os

# Data Handle
class DataHandler:
    """
    A class to load and provide access to the
    NBME clinical patient notes dataset.
    """

    def __init__(self, data_dir="../storage"):
        self.csv_path = os.path.join(data_dir, "patient_notes.csv")
        self.df = None
        self._load_data()

    def _load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df = self.df[['pn_num', 'case_num', 'pn_history']]
            print(f"Successfully loaded {len(self.df)} patient notes.")
        except FileNotFoundError:
            print(
                f"Error: Could not find patient_notes.csv at {self.csv_path}")
            print("Please make sure your data is in a '../storage' folder.")
            self.df = pd.DataFrame(columns=['pn_num', 'case_num', 'pn_history'])

    def get_note_by_id(self, pn_num):
        if self.df.empty:
            return None, "Error: DataFrame is empty. Could not load data."
        try:
            note_text = self.df.loc[self.df['pn_num'] == pn_num, 'pn_history'].iloc[0]
            return pn_num, note_text
        except IndexError:
            return f"Error: No note found with pn_num {pn_num}."

    def get_random_note(self):
        if self.df.empty:
            return -1, "Error: DataFrame is empty. Could not load data."
        random_row = self.df.sample(n=1).iloc[0]
        return int(random_row['pn_num']), random_row['pn_history']
