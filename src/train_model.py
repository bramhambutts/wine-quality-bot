from src.ingest import load_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import pickle
import logging
import sys


MODEL_FILE_NAME = 'data/saved_model.pkl'


def collect_data() -> list[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger = logging.getLogger()

    logger.info('Collecting data...')
    data = load_data()

    y = data['quality']
    X = data.drop(['quality'], axis=1)

    logger.info('Data collected, returning')
    return train_test_split(X, y, test_size=0.25, stratify=y)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model) -> None:
    logger = logging.getLogger()

    logger.info('Training model...')
    model.fit(X_train, y_train)
    logger.info('Model trained')


def test_model(X_test: pd.DataFrame, y_test: pd.Series, model):
    logger = logging.getLogger()

    logger.info('Predicting results...')
    y_pred = model.predict(X_test)

    logger.info('Determining evaluation metrics...')
    ROUND_LEVEL = 4
    logger.info(f'Accuracy score: {round(accuracy_score(y_test, y_pred), ROUND_LEVEL)}')
    logger.info(f'Precision: {round(precision_score(y_test, y_pred, average='micro'), ROUND_LEVEL)}')
    logger.info(f'Recall: {round(recall_score(y_test, y_pred, average='micro'), ROUND_LEVEL)}')
    logger.info(f'F1 score: {round(f1_score(y_test, y_pred, average='micro'), ROUND_LEVEL)}')
    logger.info(f'Confusion matrix:\n{confusion_matrix(y_test, y_pred)}')


def create_model() -> None:
    logger = logging.getLogger()

    X_train, X_test, y_train, y_test = collect_data()

    logger.info('Creating model...')
    # new_model = GaussianNB()
    # new_model = LogisticRegression(random_state=42)
    new_model = RandomForestClassifier(random_state=42)
    train_model(X_train, y_train, new_model)

    test_model(X_test, y_test, new_model)

    logger.info(f'Saving model data to {MODEL_FILE_NAME}...')
    with open(MODEL_FILE_NAME, 'wb') as file:
        pickle.dump(new_model, file)
    logger.info(f'Model saved to {MODEL_FILE_NAME}')
    

def get_model():
    logger = logging.getLogger()

    try:
        logger.info('Retrieving model...')
        with open(MODEL_FILE_NAME, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        create_model()
        logger.info('Re-retrieving model')
        with open(MODEL_FILE_NAME, 'rb') as file:
            model = pickle.load(file)
    finally:
        logger.info('Model retrieved')
        return model


if __name__ == '__main__':
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)
    create_model()