import os

from typing import Dict

PROJECT_LOGGER: Dict = {
    'sink': 'logs/development.log',
    'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {thread} | {level: <8} | '
              '{file}:{function}:{line} - {message}',
    'rotation': '50 MB',
    'retention': 10,
    'enqueue': True  # ensure logs integrity in multiprocessing environments
}

PRODUCTION_LOGGER: Dict = {
    'sink': 'logs/production.log',
    'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {thread} | {level: <8} | '
              '{file}:{function}:{line} - {message}',
    'rotation': '100 MB',
    'retention': 10,
    'enqueue': True  # ensure logs integrity in multiprocessing environments
}

CPU_THREADS: int = os.cpu_count() * 2
CPU_PROCESSES: int = round(os.cpu_count() / 2)
POST_BATCH_SIZE: int = 1000

# configs of DataAPI connection
ELK_API_KEY: str = os.getenv('ELK_API_KEY', '')
ELK_API_HOST: str = os.getenv('ELK_API_HOST', 'api.foodakai.com')
ELK_API_PORT: str = os.getenv('ELK_API_PORT', '')
