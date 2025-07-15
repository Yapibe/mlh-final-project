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

### 4. Google Cloud Authentication (Local)
To access the MIMIC-III dataset on BigQuery from a local environment, you must first authenticate. Run the following command in your terminal and follow the prompts:
```bash
gcloud auth application-default login
```
This only needs to be done once.

## Project Conventions
- All project-specific code should be located in the `project/` directory.
- The entry point for the evaluation pipeline is `project/unseen_data_evaluation.py`.
- Core logic for data processing and modeling should be in `project/pipeline.py`.
- All Python code should be formatted according to PEP 8.

## Development Workflow

We will follow a "notebook-first" hybrid approach:

1.  **Authenticate:** Begin any new session by running the `notebooks/00_setup_and_auth.ipynb` notebook to ensure you are connected to Google Cloud.
2.  **Prototype in Notebooks:** For any new task (e.g., data cleaning, feature engineering), create a new, descriptively named notebook in the `notebooks/` directory.
3.  **Experiment and Finalize:** Use the notebook to interactively develop and finalize the logic until it works correctly and is well-understood.
4.  **Refactor to Scripts:** Once the logic is stable, refactor it into a clean, reusable function within the appropriate Python script in the `project/` directory (e.g., `project/pipeline.py`).
5.  **Import and Verify:** Update the original notebook to import the new function from the script and use it. This ensures the notebook serves as a clean record of the work and validates the refactored code.
