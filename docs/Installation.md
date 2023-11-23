# Installation

## Operating System (OS)

The project is developed, tested and documented for a Unix-based distribution. Specifically, the project is being
developed under the `macOS` 12 Monterey operating system. If a non-Linux installation is needed, οne has to suitably
adapt some actions below.

## System-Wide Dependencies

The following system-wide dependencies are needed in order to have a complete development environment:


1. [git](https://git-scm.com/) (>=v2.39.2)

2. [Python](https://www.python.org/) (>=3.8, <3.9)

Also, in order to ensure proper connectivity between all components, please install these additional dependencies
below, via `brew` or `apt`:

```bash
$ apt install python3-pip  # Python package installer
$ apt install python3-dev  # header files and a static library for Python
```

## Environmental Variables

The project requires a variety of environmental variables for full functionality. For defining these
environmental variables the decision was to use the package python-dotenv. The variables should
be defined in a hidden file with the name `.env`. You should define all the environmental
variables present in the `.env.template` file with appropriate values.

**NOTE**: The variable assignments should not contain whitespaces at all.

## Getting started

Clone the project at a local workspace of your choosing with:

```bash
$ cd /where/you/want/the/project/to/exist
$ git clone git@bitbucket.org:agroknowdev/foodner.git
```

Then, at the root of the newly created folder:

```bash
$ curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -
```

This will install the project’s dependency manager, [poetry]([https://python-poetry.org](https://python-poetry.org)).

To install all the project dependencies (required and dev) that will be needed during development, simply type:

```bash
$ poetry install
$ /bin/bash build.sh
```

**NOTE**: An error `NoCompatiblePythonVersionFound` may arise if your system-wide python is not compatible with the
specification requested in the `pyproject.toml`.

In that case, use your package installer (`brew`, `apt`, `packman`, etc.) to explicitly install `python3.8` in your
system. After ensuring that this version is included in the PATH, the command `poetry env use python3.8` will
now ensure that the installation can proceed without problems.

For every other “poetry install” related error (e.g. `TooManyRedirects`), type:

```bash
$ poetry cache clear pypi --all
```

and re-install should probably solve it.

## Install Spacy Libs
```bash
$ poetry run python -m spacy download en_core_web_sm
$ poetry run python -m spacy download en_core_web_lg
```

## Run tests

Project has a comprehensive test suite available. To run all the tests, simply type:

```bash
$ poetry run pytest
```

All available tests should pass successfully.

If you want to also inspect basic test coverage metric, you can do so by:

```bash
$ poetry run pytest --cov
```

For a deeper dive (e.g. explore source code and test coverage in-browser), you can type:

```bash
poetry run pytest --cov-report html --cov-config=.coveragerc --cov
```

then, open the auto-created `./test_coverage/index.html` in your favourite browser.


---

**NOTE**: You can also run poetry commands without having to prefix the `poetry run`. All you have to do is:

```bash
$ poetry shell
```

and inside the virtual environment you can say for example:

```bash
pytest --cov
```
