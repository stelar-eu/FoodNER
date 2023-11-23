from typing import Dict, List
from unittest import TestCase

import responses
from loguru import logger

from src.etl.api.exceptions import DataAPIResponseError
from src.etl.api.interface import APIInterface


class TestInterface(TestCase):

    @classmethod
    def setUpClass(cls):
        logger.disable('src.etl.api.interface')

    def setUp(self) -> None:
        self.request = APIInterface()

    #     Test generate_text_documents method
    # ----------------------------------------------
    @responses.activate
    def test_generate_text_documents_failed_request(self) -> None:
        """Test the generate_text_documents method."""
        responses.add(
            responses.POST,
            url='http://148.251.22.254:8080/search-api-1.0/search/',
            json={'error': ['The given strictQuery or attribute has no right form.']},
            status=400,
        )

        with self.assertRaises(DataAPIResponseError):
            self.request.generate_text_documents(data_from='2018-01-01', data_until='2020-03-15', lang='en')

    @responses.activate
    def test_generate_text_documents_server_error_arises(self) -> None:
        """Test the generate_text_documents method."""
        responses.add(
            responses.POST,
            url='http://148.251.22.254:8080/search-api-1.0/search/',
            json={'errors': ['There is no apikey in body']},
            status=500,
        )

        with self.assertRaises(DataAPIResponseError):
            self.request.generate_text_documents(data_from='2018-01-01', data_until='2020-03-15', lang='en')

    @responses.activate
    def test_generate_text_documents_malformed_response(self) -> None:
        """Test the generate_text_documents method."""
        responses.add(
            responses.POST,
            url='http://148.251.22.254:8080/search-api-1.0/search/',
            json='this-ain\'t-json',
            status=200,
        )

        with self.assertRaises(TypeError):
            self.request.generate_text_documents(data_from='2018-01-01', data_until='2020-03-15', lang='en')

    @responses.activate
    def test_generate_text_documents_has_no_results(self) -> None:
        """Test the generate_text_documents method."""
        response_data: Dict = {
            'took': 26,
            'hits': {
                'total': 0,
                'max_score': 0.0,
                'hits': []
                }
        }

        responses.add(
            responses.POST, url='http://148.251.22.254:8080/search-api-1.0/search/',
            json=response_data,
            status=200,
        )

        self.assertEqual(self.request.generate_text_documents(data_from='1998-01-01', data_until='1999-03-15'), [])

    @responses.activate
    def test_generate_text_documents(self) -> None:
        """Test the generate_text_documents method."""
        response_data: Dict = {
            'took': 37,
            'hits': {
                'total': 61,
                'max_score': None,
                'hits': [
                    {
                        '_type': 'smart_scheme',
                        '_id': 'id_1',
                        '_source': {
                            'id': 'id_1',
                            'title': 'title_1',
                            'entityType': 'regulation_piece',
                            'createdOn': '2023-03-20',
                            'updatedOn': '2023-04-23T20:11:22.711972',
                            'information': {
                                'country': ['united states'],
                                'language': ['en'],
                                'descriptions': ['text_1']
                            }
                        }
                    },
                    {
                        '_type': 'smart_scheme',
                        '_id': 'id_2',
                        '_source': {
                            'id': 'id_2',
                            'title': 'title_2',
                            'entityType': 'regulation_piece',
                            'createdOn': '2023-03-20',
                            'updatedOn': '2023-04-23T20:11:22.712007',
                            'information': {
                                'country': ['europe'],
                                'language': ['en'],
                                'descriptions': ['text_2']
                            }
                        }
                    }

                ]
            }
        }

        responses.add(
            responses.POST,
            url='http://148.251.22.254:8080/search-api-1.0/search/',
            json=response_data,
            status=200,
        )

        expected: List[Dict] = [
            {'id': 'id_1', 'data': ['text_1']},
            {'id': 'id_2', 'data': ['text_2']}
        ]

        self.assertCountEqual(
            self.request.generate_text_documents(data_from='2018-01-01', data_until='2020-03-15'),
            expected)

    def tearDown(self) -> None:
        self.request = None

    @classmethod
    def tearDownClass(cls):
        logger.enable('src.etl.api.interface')
