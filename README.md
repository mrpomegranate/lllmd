
## 📦 Installation & Setup

This project uses `pyproject.toml` for dependency management.

### 1. Create & Activate Virtual Environment

```bash
python3 -m venv venv
```

### 2. Install Dependencies (Using pyproject.toml)
```bash 
source venv/bin/activate
pip install .
```

### 3. Environment Variables

Create a file named .env in the project root:
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
SERPER_API_KEY="YOUR_SERPER_API_KEY"


## 🖥️ Running the Application

With your environment active:

```bash
uv run uvicorn app:app --reload 
```
The local server starts at:
```
http://127.0.0.1:8000
```

## Demo Video

https://github.com/user-attachments/assets/19843706-de5f-4577-97db-311822c86934



