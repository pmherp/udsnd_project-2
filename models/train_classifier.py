# import libraries
#for wrangling
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
#for regular expressions
import re
#for natural language processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
#for modelling
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
#for pickeling
import pickle
#parallel backend
from sklearn.utils import parallel_backend

import sys


def load_data(database_filepath):
    """
    Load data from SQL database, split it into X, Y
    Args:
        - database_filepath: complete path to database
    Returns:
        - X: features for modeling
        - Y: true outputs for modeling
        - cat: labels of loaded data
    """
    # load data from database
    engine = create_engine(str('sqlite:///')+str(database_filepath))
    df = pd.read_sql_table('cleaned_messages', engine)
    
    #as specified in part 3
    X = df['message'].values
    Y = df.iloc[:,4:].values
    cat = df.iloc[:, 4:].columns.tolist()

    return X, Y, cat

def tokenize(text):
    """
    Normalizes text: Removes punctuation, tokenizes text, cleans tokens, makes al lowercase and removes spaces, lematizes tokens
    Args:
        - text: for countVectorizer from sklearn (Convert a collection of text documents to a matrix of token counts)
    Returns:
        - clean_tokens
    """
    # remove punctuation characters
    text = re.sub(r"[^A-Za-z0-9\-]", " ", text)
    
    #creates tokens
    tokens = word_tokenize(text)

    #remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # tag each word with part of speech
    pos_tag(tokens)
    
    #convert token to meaningful base form
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        #actually lemmatize, make all lowercase, remove space and append
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Creates pipeline of model and parameters to optimize it
    Args:
        - None
    Returns:
        - cv: grid search object
    """
    #build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3, leaf_size=30)))    
    ])

    #define gridsearch parameters
    parameters = {
        'clf__estimator__n_neighbors': [5],
        'clf__estimator__leaf_size': [20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Runs prediction on model and outputs accuracy, macro avg and weighted avg
    Args:
        - model
        - X_test: x values of test data
        - Y_Test: y values of test data
        - category_names: categories (cat) of loaded data
    Returns:
        - None
    """
    #print best accuracy of grid combinations
    print(model.best_score_)

    y_pred=model.predict(X_test)
    #print f1 score, precision
    for i in range(len(category_names)):
        print(classification_report(Y_test[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Pickles model
    Args:
        - model
        - model_filepath: complete path to saved model
    Returns:
        - None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()
        
            print('Training model...')
            model.fit(X_train, Y_train)
        
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()