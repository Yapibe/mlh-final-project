# Gemini Project Configuration

This file provides project-specific instructions for the Gemini agent.

## Environment Setup

This project uses `venv` for virtual environments and `uv` for package management.

### 1. Create Virtual Environment
If the `.venv` directory does not exist, create it:
```bash
python -m venv .venv
```

### 2. Activate Virtual Environment
- **Windows:** `.\.venv\Scripts\activate`
- **Linux/macOS:** `source .venv/bin/activate`

### 3. Install Dependencies
Dependencies are listed in `project/requirements.txt`. Install them using `uv`:
```bash
uv pip install -r project/requirements.txt
```

## Project Conventions
- All project-specific code should be located in the `project/` directory.
- The entry point for the evaluation pipeline is `project/unseen_data_evaluation.py`.
- Core logic for data processing and modeling should be in `project/pipeline.py`.
- All Python code should be formatted according to PEP 8.
