# Command Line Interface

Gathers the functionality of the FoodNER under a Command Line Interface (CLI).

## How to invoke the FoodNER CLI?

From the root directory of the project, assuming that the virtual environment is
activated, simply run `python cli.py`.

### I need the FoodNER CLI as a system command!

From the root directory of the project, assuming that the virtual environment is
activated, run `pip install -e cli`. The CLI will be installed, and the command foodner_cli should be available.

**Note:** If you want to use the foodner_cli command ensure that the virtual environment is activated.

If you use `foodner_cli` or `python cli` the following is displayed:

```bash
Usage: foodner_cli [OPTIONS] COMMAND [ARGS]...

     _____               _ _   _ _____ ____
    |  ___|__   ___   __| | \ | | ____|  _ \
    | |_ / _ \ / _ \ / _` |  \| |  _| | |_) |
    |  _| (_) | (_) | (_| | |\  | |___|  _ <
    |_|  \___/ \___/ \__,_|_| \_|_____|_| \_\

  FoodNer - Command Line Interface.

  Current functionalities:
      * Dataset: Functionalities for the manipulation of the datasets.
      * Evaluation: ...
      * Prediction: ...
      * Training: ...

Options:
  --help  Show this message and exit.

Commands:
  dataset     Prepare and analyse datasets.
  evaluation  Evaluate NER models.
  prediction  Predict Name Entities in text files.
  training    Train NLP forecasting models.
```

## How to use the CLI?

If you invoke the CLI with the `--help` option you should receive a list with the available commands.
The available commands include:

1. dataset
2. evaluation
3. prediction
4. training

Each of these commands contains a list of sub commands that are responsible for a variety of tasks.

## The *dataset* commands

Let’s start with the dataset commands. If the user writes:

`python cli.py dataset --help` the following is displayed:

Alternatively if foodner_cli is installed: `foodner_cli dataset --help`

```bash
Usage: cli.py dataset [OPTIONS] COMMAND [ARGS]...

  Prepare and analyse datasets.

Options:
  --help  Show this message and exit.

Commands:
prepare_train_dataset A command that prepares and analyze train dataset for NLP models.
prepare_inference_dataset A command that prepares and analyze inference dataset for NLP models.
```

The user will be informed about the available sub-commands. In this case the sub commands include:

### The *prepare_train_dataset* sub command

As already mentioned, the purpose of the _prepare\_train\_dataset_ command focuses on the preparation/manipulation of
the annotated data exporting data to files appropriate from training pipeline.

The _prepare\_train\_dataset_ command has the following argument:
* –data_from (-from): Starting date of the retrieved data from DataAPI. Input date format YYYY-MM-DD. Default value 2000-01-01.
* –data_until (-until): Last day of the retrieved data from DataAPI. Input date format YYYY-MM-DD. Default value today.
* –dataset_format (-dataset): The format that freetext will be exported. Default value spacy.

**Example:** `python cli.py dataset prepare-train-dataset -from 2020-01-01 -until 2023-12-12 -dataset spacy` 


### The *prepare_inference_dataset* sub command

The purpose of the _prepare\_inference\_dataset_ command focuses on the preparation/parsing of the text documents 
from DataAPI exporting data to files appropriate from NER inference.

The _prepare\_inference\_dataset_ command has the following argument:

* –data_from (-from): Starting date of the retrieved data from DataAPI. Input date format YYYY-MM-DD. Default value 2000-01-01.
* –data_until (-until): Last day of the retrieved data from DataAPI. Input date format YYYY-MM-DD. Default value today.
* –dataset_format (-format): The format that freetext will be exported. Default value spacy.

**Example:** `python cli.py dataset prepare-inference-dataset -from 2020-01-01 -until 2023-12-12 -dataset spacy` 

---

## The *training* commands

Let’s continue with the training commands. If the user writes:

`python cli.py training --help` the following is displayed:

Alternatively if foodner_cli is installed: `foodner_cli training --help`


```bash
Usage: cli.py training [OPTIONS] COMMAND [ARGS]...

  Train NER models.

Options:
  --help  Show this message and exit.

Commands:
train_ner  A command that trains a NER model on...
```

The user will be informed about the available sub-commands. In this case the sub commands include:

### The *train_ner* sub command

The purpose of the _train\_ner_ command focuses on the training of the NER models on the final dataset
according to the given model format.

The _train\_ner_ command has the following argument:

* –dataset_format (-dataset): The format that freetext will be exported. Default value spacy.
* –model_format (-model): The model which will be used to train and final NER model. Default value transformer.

**Example:** `python cli.py training train-ner -dataset spacy -model transformer`

---

## The *evaluation* commands

Let’s continue with the evaluation commands. If the user writes:

`python cli.py evaluation --help` the following is displayed:

Alternatively if prediction_cli is installed: `foodner_cli evaluation --help`

```bash
Usage: cli.py evaluation [OPTIONS] COMMAND [ARGS]...

  Evaluate NER models

Options:
  --help  Show this message and exit.

Commands:
evaluate-ner-evaluation-set  A command that trains and evaluates a NER model of the evaluation set.
```

The user will be informed about the available sub-commands. In this case the sub commands include:

### The *evaluate_ner_evaluation_set* sub command

The purpose of the _evaluate\_ner\_evaluation\_set_ command focuses on the validation of the NER
models on the **evaluation set** according to the given model format.

The _evaluate\_ner\_evaluation\_set_ command has the following argument:

* –dataset_format (-dataset): The format that freetext will be exported. Default value spacy.
* –model_format (-model): The model which will be used to train and evaluate the NER model. Default value transformer.

**Example:** `python cli.py evaluation evaluate-ner-evaluation-set -dataset spacy -model transformer`

---

## The *prediction* commands

Let’s continue with the prediction commands. If the user writes:

`python cli.py prediction --help` the following is displayed:

Alternatively if prediction_cli is installed: `foodner_cli prediction --help`

```bash
Usage: cli.py prediction [OPTIONS] COMMAND [ARGS]...

  Predict Name Entities in raw text.

Options:
  --help  Show this message and exit.

Commands:
predict_name_entities  A command that extracts the name entities from a raw text.
```

The user will be informed about the available sub-commands. In this case the sub commands include:

### The *predict_name_entities* sub command

The purpose of the _predict\_name\_entities_ command focuses on the prediction (inference) of Name Entities on the
final dataset according to the given algorithm and relevant parameters.

The _predict\_name\_entities_ command has the following argument:

* –dataset_format (-dataset): The format that freetext will be exported. Default value spacy.
* –model_format (-model): The model which will be used to train and evaluate the NER model. Default value transformer.

**Example:** `python cli.py prediction predict-name-entities -dataset spacy -model transformer`

