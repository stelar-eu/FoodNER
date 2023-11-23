from typing import Dict, List, Optional, Union

from loguru import logger

from src.etl.api.http_requests import APIRequest


class APIInterface(APIRequest):

    def __init__(self):
        super().__init__()

    def generate_text_documents(self, data_from: str, data_until: str, lang: str = 'en') -> List:
        """Provides the text documents from search request in DataAPI.

        Args:
            data_from (str): Starting date of data.
            data_until (str): Ending date of data.
            lang (str, optional): The selected language of the documents. Default value en.

        Returns:
            List
        """
        results = list()
        response: Dict = self.get_text_documents(data_from, data_until, lang)

        for bucket in self._parse_response(response):
            index_lang = bucket['_source']['information']['language'].index(lang)
            results.append({
                'id': bucket['_source']['id'],
                'data': [bucket['_source']['information']['descriptions'][index_lang]]
            })

        return results

    @staticmethod
    def _parse_response(response: Dict) -> Union[List[Optional[Dict]], int]:
        """Method to parse the given response.

        Args:
            response (Dict): The response from DataAPI.

        Returns:
            List[Dict or None] or Int

        Raises:
            KeyError: Dictionary with missing key.
        """
        try:
            return response['hits']['hits']

        except KeyError or TypeError:
            logger.warning(f'Failed to parse DataAPI response.')
            logger.info(f'Failed parsed response: {response}')
            return []
