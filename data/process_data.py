import sys
import pandas as pd 
import numpy as np 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads data and merges loaded df together
    Args:
        - messages_filepath: csv-file with correct working directory
        - categories_filepath: csv-file with correct working directory
    Returns:
        df: merged dataframe of input dataframes
    """
    #load files as pd.DataFrame
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath, delimiter=',')

    #merge files
    df = messages.merge(categories, on=['id'], how='outer')

    return df


def clean_data(df):
    """
    Cleans the given data in the merged df
    Args:
        - df: merged dataframe of load_data
    Returns:
        - df: cleaned version of merged dataframe
    """
    #create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))

    #select the first row of the categories dataframe
    row = categories.iloc[0].to_list()

    #use this row to extract a list of new column names for categories.
    category_colnames = [(lambda x: x[0:-2])(x) for x in row]

    #rename the columns of `categories`
    categories.columns = category_colnames

    #set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        #convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        #delete all rows with values 2
        categories.drop(categories[categories[column] == 2].index, inplace=True)
    
    #drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
   
    #check number of duplicates
    if df.duplicated().sum() > 0:
        #drop duplicates
        df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Saves the cleaned df to a sqlite database
    Args:
        - df: cleaned dataframe
        - database_filename: location and name of database to save df into
        - table: name of the table that is created
    Returns:
        None
    """
    #create engine with sqlalchemy
    engine = create_engine(str('sqlite:///')+str(database_filename), encoding='UTF-8')

    #save df as type sql
    df.to_sql('cleaned_messages', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()