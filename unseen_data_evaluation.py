import pandas as pd
from google.cloud import bigquery
from . import pipeline

def run_pipeline_on_unseen_data(subject_ids, client):
  """
  Run your full pipeline, from data loading to prediction.

  :param subject_ids: A list of subject IDs of an unseen test set.
  :type subject_ids: List[int]

  :param client: A BigQuery client object for accessing the MIMIC-III dataset.
  :type client: google.cloud.bigquery.client.Client

  :return: DataFrame with the following columns:
              - subject_id: Subject IDs, which in some cases can be different due to your analysis.
              - mortality_proba: Prediction probabilities for mortality.
              - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
              - readmission_proba: Prediction probabilities for readmission.
  :rtype: pandas.DataFrame
  """
  # Load the initial cohort data
  initial_cohort = pd.read_csv('../data/initial_cohort.csv')

  # Get the data for the given subject IDs
  data = pipeline.get_data(client, subject_ids)

  # Preprocess the data
  preprocessed_data = pipeline.preprocess_data(data)

  # Train the models
  mortality_model = pipeline.train_model(preprocessed_data, 'mortality')
  prolonged_los_model = pipeline.train_model(preprocessed_data, 'prolonged_los')
  readmission_model = pipeline.train_model(preprocessed_data, 'readmission')

  # TODO: Make predictions on the unseen data and return the results in the specified format.

  raise NotImplementedError('You need to implement this function')
