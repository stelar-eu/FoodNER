import click
from loguru import logger

from src.controllers.model_training import ModelTraining
from src.tools.general_tools import time_it


@click.command()
@click.option('--dataset_format',
              '-dataset',
              required=True,
              default='spacy',
              type=click.Choice(['spacy', 'pytorch', 'bert'], case_sensitive=False),
              help='The format that freetext will be exported. Default value spacy')
@click.option('--model_format',
              '-model',
              required=True,
              default='transition',
              type=click.Choice(['transition', 'transition_simple', 'transformer', 'bert'], case_sensitive=False),
              help='The model which will be used to train the NER model. Default value Transition.')
@time_it
def train_ner(dataset_format: str, model_format: str) -> None:
    """A command that trains a NER model on the total dataset.

    \b
    dataset_format (str): The format that freetext will be exported. Default value spacy.

    \b
    model_format (str): The model which will be used to train the NER model. Default value transition.

    """
    try:
        ModelTraining(dataset_format, model_format).train_ner()

    except Exception as e:
        logger.critical('NER training failed to be completed: {0}'.format(str(e)))
