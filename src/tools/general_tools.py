import json
import os
import pickle
from datetime import timedelta
from functools import wraps
from time import time
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

from loguru import logger


def get_folder_path(path_from_module: str) -> str:
    """Method to find the folders that in many cases is needed but are not visible.

    Args:
        path_from_module (str): The path from the central repo to the folder.

    Returns:
        str, the actual path to directory
    """
    fn = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))).split('src')[0]
    return '{0}{1}'.format(fn, path_from_module)


def get_filepath(path_from_module: str, filename: str) -> str:
    """Method to find the path-files that in many cases is needed but are not visible.

    Args:
        path_from_module (str): The path from the central repo to the folder.
        filename (str): The file we want from the folder.

    Returns:
        str, the actual path to file
    """
    filename = filename.replace('/', ':')
    fn = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))).split('src')[0]
    return '{0}{1}/{2}'.format(fn, path_from_module, filename)


def is_number(value: Any) -> bool:
    """Method to check if a given input is number or not.

    Args:
        value (any): The given input.

    Returns:
        bool
    """
    try:
        float(value)
        return True

    except (ValueError, TypeError):
        return False


def is_empty(value: Any) -> bool:
    """Method to check if a given input is empty.

    Args:
        value (any): The given input.

    Returns:
        bool
    """
    return not bool(value)


def extract_info_from_url(url: str) -> Dict:
    """Method to extract protocol, hostname, path, params, query, username, password and port from url given.

    Args:
        url (str): The site url.

    Returns:
        Dict
    """
    obj = urlparse(url)
    return {'protocol': obj.scheme, 'hostname': obj.hostname, 'path': obj.path, 'params': obj.params,
            'query': obj.query, 'username': obj.username, 'password': obj.password, 'port': obj.port}


def time_it(method: Any):  # pragma: no cover
    """Print the runtime of the decorated method."""

    # required for time_it to also work with other decorators.
    @wraps(method)
    def timed(*args, **kwargs):
        start = time()
        result = method(*args, **kwargs)
        finish = time()

        logger.success(f'Execution completed in {timedelta(seconds=round(finish - start))} s. '
                       f'[method: <{method.__name__}>]')
        return result

    return timed


def load_pickled_data(path: str):  # pragma: no cover
    """Load data from pickle.

    Args:
        path (str): The path to file.

    Returns:
        the pickled data
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        return data

    except FileNotFoundError:
        logger.warning(f'File/Dir {path} is not found.')


def dump_pickled_data(path: str, data: object) -> None:  # pragma: no cover
    """Store given data to pickle.

    Args:
        path (str): The path to file.
        data (object): The data to be dumped as pickle.

    Returns:
        None
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    except FileNotFoundError:
        logger.warning(f'File/Dir {path} is not found.')


def load_json_data(path: str) -> List:
    """Method reads and loads the data of a JSON file.

    Args:
        path (str): The JSON filepath to read.

    Returns:
        Dict
    """
    try:
        with open(path, 'r') as json_file:
            return [json.loads(line) for line in json_file]

    except FileNotFoundError:
        raise FileNotFoundError(f'JSON file {path} is not found.')


def dump_json_data(path: str, data: Union[Dict, List], indent: int = 4, lines: bool = False) \
        -> None:  # pragma: no cover
    """Store given data to a json or jsonl file.

    Args:
        path (str): The path to file.
        data (dict or list): The data to be dumped as JSON file.
        indent (int, optional): The indent value. Default value 4.
        lines (bool, optional): Create .json or jsonl file.

    Returns:
        None
    """
    try:
        with open(path, 'w') as fout:

            if lines:  # case of jsonl file
                for item in data:
                    fout.write(f'{json.dumps(item)}\n')
            else:
                fout.write(json.dumps(data, indent=indent))

    except FileNotFoundError:
        raise FileNotFoundError(f'File/Dir {path} is not found.')
