import click
from loguru import logger

from src.controllers.prediction import ModelPredict
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
def predict_name_entities(dataset_format: str, model_format: str) -> None:
    """A command that extracts the name entities from a raw text.

    \b
    dataset_format (str): The format that freetext will be exported. Default value spacy.

    \b
    model_format (str): The model which will be used to train the NER model. Default value transition.

    """
    try:
        ModelPredict(dataset_format, model_format).predict()

    except Exception as e:
        logger.critical('Prediction of name entities failed to be completed: {0}'.format(str(e)))
