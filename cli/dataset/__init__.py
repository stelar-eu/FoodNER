import click

from cli.dataset.dataset_commands import prepare_train_dataset, prepare_inference_dataset


@click.group('dataset')
def dataset() -> None:
    """Prepare and analyse datasets."""


dataset.add_command(prepare_inference_dataset)
dataset.add_command(prepare_train_dataset)
