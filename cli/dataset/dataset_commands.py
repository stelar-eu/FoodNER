from datetime import datetime

import click
from loguru import logger

from src.controllers.data_preparation import DataPreparation
from src.tools.general_tools import time_it


@click.command()
@click.option('--data_from',
              '-from',
              required=False,
              default='2000-01-01',
              type=click.STRING,
              help='Starting date of the retrieved data from DataAPI. Input date format YYYY-MM-DD.'
                   'Default value 2000-01-01.')
@click.option('--data_until',
              '-until',
              required=False,
              default=datetime.now().strftime('%Y-%m-%d'),
              type=click.STRING,
              help='Last day of the retrieved data from DataAPI. Input date format YYYY-MM-DD. Default value today.')
@click.option('--dataset_format',
              '-dataset',
              required=True,
              default='spacy',
              type=click.Choice(['spacy', 'pytorch', 'bert'], case_sensitive=False),
              help='The format that freetext will be exported. Default value spacy.')
@time_it
def prepare_train_dataset(data_from: str, data_until: str, dataset_format: str) -> None:
    """A command that prepares and analyze train dataset for NLP models.

    \b
    data_from (str): Starting date of the retrieved data from DataAPI.
                    Input date format YYYY-MM-DD. Default value 2000-01-01.

    \b
    data_until (str): Last day of the retrieved data from DataAPI.
                      Input date format YYYY-MM-DD. Default value today.

    \b
    dataset_format (str): The format that freetext will be exported. Default value spacy.

    """
    # TODO: figure out data_from and data_until, there is no use of them yet
    try:
        DataPreparation(data_from, data_until, dataset_format).prepare_train_dataset()

    except Exception as e:
        logger.critical('Data preparation for train dataset failed to be completed: {0}'.format(str(e)))


@click.command()
@click.option('--data_from',
              '-from',
              required=False,
              default='2000-01-01',
              type=click.STRING,
              help='Starting date of the retrieved data from DataAPI. Input date format YYYY-MM-DD.'
                   'Default value 2000-01-01.')
@click.option('--data_until',
              '-until',
              required=False,
              default=datetime.now().strftime('%Y-%m-%d'),
              type=click.STRING,
              help='Last day of the retrieved data from DataAPI. Input date format YYYY-MM-DD. Default value today.')
@click.option('--dataset_format',
              '-dataset',
              required=True,
              default='spacy',
              type=click.Choice(['spacy', 'pytorch'], case_sensitive=False),
              help='The format that freetext will be exported. Default value spacy.')
@time_it
def prepare_inference_dataset(data_from: str, data_until: str, dataset_format: str) -> None:
    """A command that prepares and analyze inference dataset for NLP models.

    \b
    data_from (str): Starting date of the retrieved data from DataAPI.
                    Input date format YYYY-MM-DD. Default value 2000-01-01.

    \b
    data_until (str): Last day of the retrieved data from DataAPI.
                      Input date format YYYY-MM-DD. Default value today.

    \b
    dataset_format (str): The format that freetext will be exported. Default value spacy.

    """
    try:
        DataPreparation(data_from, data_until, dataset_format).prepare_inference_dataset()

    except Exception as e:
        logger.critical('Data preparation for inference dataset failed to be completed: {0}'.format(str(e)))
