import click

from cli.training.train_commands import train_ner


@click.group('training')
def training() -> None:
    """Train NER models."""
    pass


training.add_command(train_ner)
