import click

from cli.prediction.prediction_commands import predict_name_entities


@click.group('prediction')
def prediction() -> None:
    """Predict Name Entities in raw text."""
    pass


prediction.add_command(predict_name_entities)
