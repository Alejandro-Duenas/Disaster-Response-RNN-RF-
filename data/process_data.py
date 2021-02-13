#----------------------------- Process Data File--------------------------------

# 1. Import libraries:

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


# 2. Define basic functions:

def load_data(messages_filepath, categories_filepath):
    '''This function loads the datasets with the messages and categories
    Inputs:
        messages_filepath: str with the file path of the CSV file containing the 
        messages data
        categories_filepath: str with the file ppath of the CSV file containing
        the categories data 
    Outputs:
        pandas Dataframe       
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages,categories


def clean_data(df):
    '''Cleans data of the input dataframe, eliminating duplicate values
    Input:
        df: Pandas DataFrame instance
    Output:
        df: Pandas DataFrame instance (withoug duplicates)
    '''
    # Remove duplicates
    df.drop_duplicates(subset='message',keep='first',inplace=True)
    return df


def save_data(df, database_filename='message_data'):
    '''This function saves the df dataframe into a SQL file
    Inputs:
        df: Pandas DataFrame instance
        database_filename: string containing the name of the SQL table saved
        with a default value "message_data"
    Outputs:
        None
    '''
    engine = create_engine('sqlite:///{}.db'/forma(database_filename)) 
    df.to_sql(database_filename,engine,index=False)


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