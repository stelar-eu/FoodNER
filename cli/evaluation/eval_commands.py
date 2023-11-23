import click
from loguru import logger

from src.controllers.evaluation import ModelEvaluation
from src.tools.general_tools import time_it


@click.command()
@click.option('--dataset_format',
              '-dataset',
              required=True,
              default='spacy',
              type=click.Choice(['spacy', 'pytorch', 'bert'], case_sensitive=False),
              help='The format that freetext will be exported. Default value spacy.')
@click.option('--model_format',
              '-model',
              required=True,
              default='transition',
              type=click.Choice(['transition', 'transition_simple', 'transformer', 'bert'], case_sensitive=False),
              help='The model which will be used to train and evaluate the NER model. Default value RNN.')
@time_it
def evaluate_ner_evaluation_set(dataset_format: str, model_format: str) -> None:
    """A command that trains and evaluates a NER model of the evaluation set.

    \b
    dataset_format (str): The format that freetext will be exported. Default value spacy.

    \b
    model_format (str): The model which will be used to train and evaluate the NER model. Default value transition.

    """
    try:
        ModelEvaluation(dataset_format, model_format).evaluate_ner_evaluation_set()

    except Exception as e:
        logger.critical('NER evaluation for train dataset failed to be completed: {0}'.format(str(e)))
