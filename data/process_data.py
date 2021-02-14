#----------------------------- Process Data File--------------------------------

# 1. Import libraries:

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


# 2. Define basic functions:

def load_data(messages_path,categories_path):
    '''This function loads and merge the data
    Inputs:
        messages_path, categories_path: both are string that give the path
        to the CSV files that contain the information of the messages and 
        categories.
    Outputs:
        df: Pandas DataFrame that contains the merged infomation of messages
        and categoires
    '''
    # Load the data
    categories = pd.read_csv(categories_path)
    messages = pd.read_csv(messages_path)

    # Merge the data
    df = messages.merge(categories,'inner',on='id')

    # Create dataframe for the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # Create names for the categories
    raw = categories.loc[0,:]
    category_names = raw.str.slice(stop=-2)

    # Changing the names:
    categories.columns = category_names

    # Giving correct values to the category columns
    for column in categories:
        categories[column] = categories[column].astype(str).str.\
                                slice(start=-1).astype(int)

    # Join and replace the categories dataframe in the main dataframe
    df.drop(columns='categories',inplace=True) # eliminate bad column
    df = pd.concat([df,categories],axis=1,sort=False)

    return df


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
    engine = create_engine('sqlite:///{}'.format(database_filename)) 
    df.to_sql('DisasterResponse',engine,index=False)


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