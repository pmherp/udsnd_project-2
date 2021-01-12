# Disaster Response Pipeline

## Installation

The code should run with no issues using Python versions 3.*.

dependencies (used extensions/frameworks):
- pandas for data wrangling
- plotly for python vizualisations
- flask for back-end of web-app
- bootstrap for front-end of web-app
- requests for handling API connection
- sqlalchemy for creating database and table
- sklearn for training model
- nltk for handling text data
- re for regular expressions
- pickle for storing model

## Project Motivation
For this project, I was using data from Figure Eight containing labelled disaster messages with the goal to train a classification model to classify text messages into 36 given categories.

## File Descriptions
- process_data.py: preprocesses the given dataset and stores the cleaned data in a database
- train_classifier.py: takes cleaned data and trains classifier with it. stores trained model in pickled file
- run.py: loads cleaned data and trained model, plots graphs with it, manages backend of web app
- master.html: homepage of web app
- go.html. result page for running new messages on model, shows categories the input messages belongs to according to model

## Running the Code
Clone this repository and run the following commands to set up the database and the classifier:
- database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- model: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

To run the web app on your local machine, run the following command:
web app: `python run.py`
Then go to your browser and enter http://localhost:3001/ to see the web app

## Licensing, Authors, Acknowledgements
Udacity provided the file templates and code snippets that were used in the web app. 
Figure Eight provided the dataset that was used to train the classifier.
