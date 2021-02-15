# 1. Import Packages
import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
#import xgboost
from sklearn.externals import joblib
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def load_data(filepath):
    '''Load the data
    Inputs:
        filepath: string with the path of the file
    Outputs:
        X, Y: Pandas dataframes containing the predictive variables and the
        objective variable correspondingly
        category_names: list with names of the objective variables (Ys)
    '''
    engine = create_engine('sqlite:///{}'.format(filepath))
    df = pd.read_sql_table('DisasterResponse',engine)
    Y = df.loc[:,'related':'direct_report']
    category_names = Y.columns
    X = df['message']
    return X,Y,category_names

def tokenize(text):
    '''This function tokenizes the input text, using a lemmatizer
    Inputs:
        text: strin
    Outputs:
        clean_tokens: tokenized text
    '''
    # Create the tokens and the objects needed for the operation
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []

    # Cleaning the tokens: lower case, eliminate spaces and use lemmatizer
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token.lower().strip())
        clean_tokens.append(clean_token)
    return clean_tokens

class MultiModel(BaseEstimator):
    '''This class is done to select among different models inside a GridSearch
    object

    This idea was inspired by:
    https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python
    '''
    def __init__(self,estimator=RandomForestClassifier()):
        self.estimator = estimator
    
    def fit(self,x_train,y_train, **kwargs):
        self.estimator.fit(x_train,y_train)
        return self
    
    def predict(self, x_test):
        return self.estimator.predict(x_test)


def build_model():
    '''This function builds a pipeline that generates a the best models possible
    given the options explored (Random Forests, XGBoost)
    '''
    # Set the pipeline: use the class so that I could test different models
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Set different list of parameters to test in the grid search cross-validation
    parameters = [
    #{'clf__estimator':[xgboost.XGBClassifier()],'vect__max_df': (0.5, 0.75, 1.0)},
    {'tfidf__use_idf': (True, False),
    'vect__max_df': (0.5, 0.75, 1.0)}
    ]

    # Test the different combinations of hyperparameters and return best estimator
    cv = GridSearchCV(pipeline,param_grid=parameters,verbose=1)
    return cv


def evaluate_model(model,x_test,y_test,category_names):
    '''This function evaluates the model in the X and Y datasets. It prints
    a summary of the tests for all the categories in the objective dataset Y

    Inputs:
        model: a model where the results are evaluated
        x: dataset with the input varibles
        y: dataset with the target variables
        category_names: string with the names of the objective variables
    Outputs:
        None
    '''
    y_pred = model.predict(x_test)
    print(classification_report(y_test,y_pred,zero_division=0,target_names=\
                                category_names))
                                

def save_model(model,model_filepath):
    '''This function saves the model in a Pickle file'''
    joblib.dump(model,model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
