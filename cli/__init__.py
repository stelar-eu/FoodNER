import click
from loguru import logger

from config.project import PROJECT_LOGGER
from cli.dataset import dataset
from cli.evaluation import evaluation
from cli.prediction import prediction
from cli.training import training


@click.group()
def foodner_cli() -> None:
    """
    \b
     _____               _ _   _ _____ ____
    |  ___|__   ___   __| | \ | | ____|  _ \
    | |_ / _ \ / _ \ / _` |  \| |  _| | |_) |
    |  _| (_) | (_) | (_| | |\  | |___|  _ <
    |_|  \___/ \___/ \__,_|_| \_|_____|_| \_\

    \b
    FoodNER Service - Command Line Interface.

    \b
    Current functionalities:
        * Dataset: Functionalities for the manipulation of the datasets.
        * Evaluation: Functionalities for the train and evaluation of the NER models.
        * Prediction: Functionalities for the NER inference of a raw text.
        * Training: Functionalities for the train of the final NER model.
    """
    # initialize the logger
    logger.add(**PROJECT_LOGGER)


foodner_cli.add_command(dataset)
foodner_cli.add_command(evaluation)
foodner_cli.add_command(prediction)
foodner_cli.add_command(training)
