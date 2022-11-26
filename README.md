# Disaster Response Pipeline

Starting November 28th, 2022, free dynos will no longer be available. This project was on a free plan.

## Project Motivation
It is usually during crisis or disaster, that first respondants have the least capacity to go through all the text messages they receive. This application allows for them to easily cluster incoming messages into 36 categories and react appropriatly to them.

For this project, I was using data from Figure Eight containing labelled disaster messages with the goal to train a model to classify text messages into 36 given categories. 

The project is devided into three sections:
1. __Data Processing:__ an ETL pipeline to extract the data from the source, clean it and save it in a proper database structure
2. __Machine Learning Pipeline:__ trains a model to classify text messages into 36 categories
3. __Web App:__ that shows visualizations derived from the cleaned data and assigned categories to entered text messages

## Getting Started
The code should run with no issues using Python versions 3.*.

### Dependencies
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

### File Descriptions
- process_data.py: preprocesses the given dataset and stores the cleaned data in a database
- train_classifier.py: takes cleaned data and trains classifier with it. stores trained model in pickled file
- run.py: loads cleaned data and trained model, plots graphs with it, manages backend of web app
- master.html: homepage of web app
- go.html. result page for running new messages on model, shows categories the input messages belongs to according to model

### Installation
Clone this Git repository:
`git clone https://github.com/pmherp/udacity_data-scientist_project-2`

### Running the Code
1. Run the following commands in the project's root directory to set up your database and model.
    - run the ETL pipeline: 
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - run the ML pipeline: 
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app: `python run.py`
3. Go to you browser and enter http://localhost:3001/

## Author
Philip M. Herp

## Licensing, Authors, Acknowledgements
[Udacity](https://www.udacity.com/) provided the file templates and code snippets that were used in the web app. 
[Figure Eight](https://appen.com/) provided the dataset that was used to train the classifier.
