# MLH Final Project

This repository contains code and notes for the Machine Learning for Health Care (MLHC) final project. The goal is to extract a cohort from the MIMIC-III dataset and build prediction models for three clinical targets:

1. Mortality (in-hospital or within 30 days of discharge)
2. Prolonged length of stay
3. 30-day readmission

Most of the logic lives in the [notebooks](notebooks/) folder. The `skeleton.ipynb` notebook provides project setup and Google Cloud authentication instructions for accessing the MIMIC-III dataset via **BigQuery**.

## Local Requirements

The project reads data directly from BigQuery. If you run the notebooks locally (outside Google Colab), install the Google Cloud SDK and authenticate with:

```bash
gcloud auth application-default login
```

## Dependency Management

This project uses [uv](https://github.com/astral-sh/uv) for package management.

* **Install dependencies** – run `uv sync` in the repository root. This installs the packages listed in `pyproject.toml` using the locked versions from `uv.lock`.
* **Add a new dependency** – run `uv add PACKAGE_NAME` which updates both `pyproject.toml` and `uv.lock`.

## Repository Layout

- `notebooks/skeleton.ipynb` – project setup and authentication guide for Google Cloud/BigQuery access.
- `project/` – Python package with code for running the pipeline on new data.

See `project/README.md` for more details on the pipeline.
