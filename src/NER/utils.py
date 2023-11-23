from datetime import datetime
from typing import List

from src.etl.api.http_requests import APIRequest
from src.schemas.datatypes import RawSpacyData


class PredictionUtils:

    def __init__(self, doc_id: str, text: List, predictions: List[RawSpacyData]) -> None:
        """Utils for the predictions with prophet.

        Args:
            doc_id (str): The id of the document.
            text (List): A list with the tokens.
            predictions (List[RawSpacyData]): The prediction of name entities for the raw text.
        """
        self.doc_id: str = doc_id
        self.text: List[str] = text
        self.predictions: List[RawSpacyData] = predictions

    def parse_and_post_results(self) -> None:  # pragma: no cover
        """Parse and post the results on DataAPI.

        Returns:
            None
        """
        results = []

        # parse results prepare request
        for text, prediction in zip(self.text, self.predictions):

            results.append(
                {
                    'createdOn': datetime.now().isoformat(),
                    'dataSource': 'FOODAKAI',
                    'description': None,
                    'entityType': 'regulation_ner',
                    'id': f'NER_{self.doc_id}',
                    'information': {
                        'automatic_extraction':
                            {
                                'data': text,
                                'label': [
                                    {
                                        'start': ent['start'],
                                        'end': ent['end'],
                                        'label': ent['label'],
                                        'text': ent['text']
                                    }
                                    for ent in prediction
                                ]
                            }
                    },
                    'published': True,
                    'tags': [
                        'ner', 'automatic_extraction'
                    ],
                    'title': f'NER results for document {self.doc_id}.',
                    'updatedOn': datetime.now().isoformat()
                }
            )

        # post results
        APIRequest().post_predicted_results(results)
