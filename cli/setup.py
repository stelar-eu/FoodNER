from setuptools import setup

"""
Setup definition for the CLI.
"""

setup(
    name='foodner-cli',
    version='0.1',
    author='Agroknow IKE Company - Data Science Team',
    description='FoodNER CLI.',
    python_requires='>=3.8,<3.10',
    install_requires=['Click'],
    entry_points={'console_scripts': ['foodner_cli=cli:foodner_cli']},
)
