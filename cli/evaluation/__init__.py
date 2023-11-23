import click

from cli.evaluation.eval_commands import evaluate_ner_evaluation_set


@click.group('evaluation')
def evaluation() -> None:
    """Evaluate NER models."""
    pass


evaluation.add_command(evaluate_ner_evaluation_set)
