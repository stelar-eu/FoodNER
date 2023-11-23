Requirements
============

We follow the
[dependency specification system](https://python-poetry.org/docs/dependency-specification/#dependency-specification)
of `poetry`.

Packages
--------

The following are required in order for the application to run successfully:

**Package**   | **License** | **Version**
-----------   |-------------| -----------
click               | BSD         | \>=8.1.3 <9.0
dataset             | Apache 2.0  | \>=2.11.0 <2.12
loguru              | MIT         | \>=0.7.0 <0.8.0
nervaluate          | GNU GPL v3  | \>=0.1.8 <0.2.0
nltk                | Apache 2.0  | \>=3.8.1 <3.9.0
pytest              | MIT         | \>=7.3.1 <8.0.0
python-dotenv       | BSD         | \>=1.0.0 <2.0.0
requests            | Apache 2.0  | \>=2.29.0 <2.30.0
responses           | Apache 2.0  | <0.19
spacy               | MIT         | \>=3.5.2 <4.0.0
spacy-transformers  | MIT         | \>=1.2.3 <1.3.0
transformers        | Apache 2.0  | \>=4.26.1 <4.27.0
torch               | BSD         | \>=1.13.1 <1.14.0
tqdm                | MIT         | \>=4.65.0 <5.0.0

Dev-packages
------------

The following are non-essential packages to the application, that however ease, automate and simplify the development
lifecycle:

**Package** | **License** | **Version**
------------ | ------------- | -------------
coverage | Apache 2.0 | \>=7.2.5 <8.0.0
coverage-badge | MIT | \>=1.1.0 <2.0.0
pytest-cov | MIT | \>=4.0.0 <5.0.0

