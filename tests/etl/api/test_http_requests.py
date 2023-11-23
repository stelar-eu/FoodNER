import json
from unittest import TestCase
from typing import Dict, List

import responses
from loguru import logger

from src.etl.api.http_requests import APIRequest


class TestHttpRequests(TestCase):

    @classmethod
    def setUpClass(cls):
        logger.disable('src.etl.api.http_request')

    def setUp(self) -> None:
        self.request = APIRequest()

    def assertProperRequestCall(self) -> None:
        """Custom assertion method."""
        # assert that a request was made
        self.assertEqual(len(responses.calls), 1)
        # with the proper headers
        self.assertEqual('application/json', responses.calls[0].request.headers['Content-Type'])
        self.assertEqual('application/json', responses.calls[0].request.headers['Accept'])
        # with apikey in the body of the request
        self.assertIn('apikey', json.loads(responses.calls[0].request.body))

    def assertProperPostCall(self) -> None:
        """Custom assertion method."""
        # assert that a request was made
        self.assertEqual(len(responses.calls), 1)
        # with the proper headers
        self.assertEqual('application/json', responses.calls[0].request.headers['Content-Type'])
        self.assertEqual('application/json', responses.calls[0].request.headers['Accept'])
        # with the created_on in the body of the request
        self.assertIn('createdOn', json.loads(responses.calls[0].request.body)[0])
        # with the information in the body of the request
        self.assertIn('information', json.loads(responses.calls[0].request.body)[0])
        # with the information in the body of the request
        self.assertIn('dataSource', json.loads(responses.calls[0].request.body)[0])
        # with the status code in the response
        self.assertEqual(200, responses.calls[0].response.status_code)
        # with Message in the response
        self.assertIn('Message', responses.calls[0].response.json())

    @responses.activate
    def test_get_text_documents(self) -> None:
        """Test the get_text_documents method."""
        responses.add(
            responses.POST,
            url='http://148.251.22.254:8080/search-api-1.0/search/',
            json={'hits': 'returned product categories'},
            status=200,
        )

        self.request.get_text_documents(data_from='2018-01-01', data_until='2020-03-15', lang='en')
        self.assertProperRequestCall()

    @responses.activate
    def test_post_predicted_results(self) -> None:
        """Test the post_predicted_results method."""
        responses.add(
            responses.PUT,
            url='http://148.251.22.254:8080/schema-api-1.0/entity/smart-scheme/v2/mass-create',
            json={'Message': 'Predicted recalls are stored!'},
            status=200,
        )

        predictions: List[Dict] = [
            {
                'createdOn': '2022-11-01',
                'dataSource': 'FOODAKAI',
                'description': None,
                'entityType': 'prediction_trend',
                'id': 'my_id_1',
                'information': {
                    'hazard': 'NONE', 'product': 'fats and oils', 'origin': 'NONE',
                    'accuracy': 0.8, 'mae': 2, 'mape': 3, 'mse': 0.4, 'r2': 5, 'rmse': 0.6, 'actual_value': 4,
                    'tendency': 23, 'interval': 'MONTH'
                },
                'published': True,
                'tags': ['prediction_trend'],
                'title': 'A Random Title',
                'updatedOn': '2022-11-08T14:09:34.944791'
            },
            {
                'createdOn': '2022-10-01',
                'dataSource': 'FOODAKAI',
                'description': None,
                'entityType': 'prediction_trend',
                'id': 'my_id_2',
                'information': {
                    'hazard': 'migration', 'product': 'fats and oils', 'origin': 'NONE',
                    'accuracy': 0.0, 'mae': 12, 'mape': 13, 'mse': 0.8, 'r2': 15, 'rmse': 1.6, 'actual_value': 0.2,
                    'tendency': 103, 'interval': 'WEEK'
                },
                'published': True,
                'tags': ['prediction_trend'],
                'title': 'A Random Title',
                'updatedOn': '2022-10-18T12:59:16.177441'
            },
        ]

        self.request.post_predicted_results(predictions)
        self.assertProperPostCall()

    @classmethod
    def tearDownClass(cls):
        logger.enable('src.etl.api.http_request')

    def tearDown(self) -> None:
        self.request = None
