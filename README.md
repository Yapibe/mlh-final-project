# Multi-Task GRU Autoencoder for ICU Outcome Prediction

This repository implements a **novel neural architecture** for predicting ICU outcomes using MIMIC-III data. Our approach combines multi-task learning, sequence modeling, and contrastive learning to achieve state-of-the-art performance on three critical clinical targets:

1. **Mortality** (in-hospital or within 30 days of discharge)
2. **Prolonged length of stay** (>7 days)
3. **30-day hospital readmission**

## Key Innovation

Our **Multi-Task Sequence GRU Autoencoder** (`MultiTaskSeqGRUAE`) represents a significant advancement in healthcare ML:

- **🧠 Shared Patient Representation**: GRU encoder compresses 48-hour patient timeline into a rich latent embedding
- **🎯 Multi-Task Classification**: Three task-specific heads predict outcomes from shared patient representation  
- **🔄 Self-Supervised Learning**: Autoencoder reconstruction ensures robust feature learning
- **📊 Contrastive Learning**: SupCon loss clusters similar patients, with adaptive anchoring for rare events
- **⚖️ Smart Class Balancing**: Custom batch sampler ensures adequate representation of rare positive cases

## Advanced Training Strategy

```python
# Three-loss training system:
total_loss = λ_recon * reconstruction_loss +     # Learn robust representations
             λ_bce * classification_loss +       # Supervised prediction
             λ_supcon * contrastive_loss         # Similar patients cluster together
```

## Data Access

**Important**: Due to Google's policy changes, MIMIC-III data is now accessed via a **DuckDB database** on Google Drive (not BigQuery).

- **Production pipeline**: `preprocessing/preprocess_training_data.ipynb` - streamlined training data preparation
- **Exploratory analysis**: `preprocessing/exploratory_data_analysis.ipynb` - comprehensive EDA with modern DuckDB approach

## Local Requirements

The project reads data from a **DuckDB database** on Google Drive. To run locally:

1. **Create Google Drive shortcut** to the MIMIC-III database directory
2. **Mount Google Drive** in Colab (handled automatically in notebooks)
3. **Update paths** in `preprocessing/config.py` if needed

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Option 1: Using uv (Recommended)

If you don't have uv installed:
```bash
# Install uv (cross-platform)
pip install uv
# OR on macOS/Linux with curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the environment:
```bash
# Clone the repository
git clone https://github.com/Yapibe/mlh-final-project
cd MLH

# Install all dependencies (creates virtual environment automatically)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows
```

### Option 2: Using pip/conda

If you prefer traditional tools, you can also use the `pyproject.toml` file:
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Repository Layout

- `preprocessing/` – complete data pipeline with modern DuckDB approach
  - `exploratory_data_analysis.ipynb` – comprehensive EDA and data quality analysis
  - `preprocess_training_data.ipynb` – production training data preparation
  - `preprocess_pipeline.py` – core preprocessing functions
- `model/` – sophisticated multi-task neural network implementation
- `unseen_data_evaluation.py` – evaluation script for unseen test data
- `pyproject.toml` & `uv.lock` – dependency management files

## Project Structure

This project implements a multi-task learning approach for predicting three ICU outcomes:

1. **Data Pipeline** (`preprocessing/`):
   - Extracts demographics, vitals, labs, medications, and microbiology from MIMIC-III
   - Implements 48-hour prediction window with 6-hour gap
   - Handles data imputation and feature engineering

2. **Model** (`model/`):
   - Multi-task GRU autoencoder with contrastive learning
   - Three prediction heads for mortality, prolonged stay, and readmission
   - Advanced loss combination: reconstruction + classification + contrastive

3. **Evaluation**:
   - ROC/PR curves, calibration analysis, feature importance
   - Designed for evaluation on unseen test data
