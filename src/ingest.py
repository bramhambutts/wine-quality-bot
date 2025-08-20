from ucimlrepo import fetch_ucirepo
import pandas as pd
import logging
import pathlib


DATA_FILE_LOCATION = 'data/wine_quality.csv'


def download_data() -> pd.DataFrame:
    logger = logging.getLogger()

    logger.info('Downloading data...')
    wine_quality = fetch_ucirepo(id=186)
    data: pd.DataFrame = wine_quality.data.original
    logger.info('Data downloaded')

    data = clean_data(data)

    logger.info(f'Saving data to {DATA_FILE_LOCATION}...')
    try:
        data.to_csv(DATA_FILE_LOCATION, index=False)
    except OSError:
        logger.debug('Creating data directory')
        pathlib.Path('data').mkdir()
        data.to_csv(DATA_FILE_LOCATION, index=False)
    finally:
        logger.info('Data saved')


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger()
    
    logger.info('Cleaning data...')
    red_mask = data['color']=='red'
    new_data = data.drop(['color'], axis=1)
    new_data['color_code'] = red_mask.astype(int)
    logger.info('Data cleaned')
    return new_data


def load_data() -> pd.DataFrame:
    logger = logging.getLogger()

    logger.info('Loading data...')
    try:
        return pd.read_csv(DATA_FILE_LOCATION)
    except FileNotFoundError:
        download_data()
        return pd.read_csv(DATA_FILE_LOCATION)
    finally:
        logger.info('Data loaded')


if __name__ == '__main__':
    download_data()