import json
from typing import Dict, List
from datetime import datetime, date

import requests
from urllib.parse import urljoin
from urllib3.util.retry import Retry

from config.project import ELK_API_KEY, ELK_API_HOST, ELK_API_PORT
from src.etl.api import adapters
from src.etl.api.session import DataAPISession


class APIRequest:
    """Class that handles the requests and the interaction with the DataAPI."""

    # static attributes for base url and endpoints
    API_BASE: str = f'http://{ELK_API_HOST}:{ELK_API_PORT}'

    def __init__(self,
                 apikey: str = ELK_API_KEY,
                 base_url: str = API_BASE,
                 timeout: int = 30,
                 max_retries: int = 3,) -> None:
        """Creates a session with Data API for receiving and posting data.

        Returns:
            None
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.apikey = apikey.strip()
        self._headers: Dict = self._get_headers()
        self.session: requests.Session = self._setup_session()

    def get_text_documents(self, data_from: str = None, data_until: str = None, lang: str = 'en'):
        """Retrieves product categories from the Data API.

        Args:
            data_from (str, optional): Starting date of data. Default value None.
            data_until (str, optional): Ending date of data. Default value None.
            lang (str, optional): The selected language of the documents. Default value en.

        Returns:
            Dict
        """
        request: Dict = {
            'apikey': self.apikey,
            'strictQuery': {
                'information.language': lang
            },
            'pageSize': 100,
            'from': data_from if data_from else '2000-01-01',
            'to': data_until if data_until else datetime.now().strftime('%Y-%m-%d'),
            'entityType': 'regulation_piece',
            'detail': True,
            'page': 0
        }

        # construct the url of the endpoint
        url = self._build_url('search-api-1.0/search/')

        return self.session.post(
            url=url,
            json=request,
            headers=self._headers,
        )

    def post_predicted_results(self, predictions: List[Dict]) -> requests.models.Response:
        """Post predicted recalls on Data API.

        Args:
            predictions (List[Dict]): List with the predicted values and results.

        Returns:
            requests.models.Response
        """
        def default(o) -> str:
            if isinstance(o, (date, datetime)):
                return o.isoformat()

        # construct the url of the endpoint
        url = self._build_url(f'schema-api-1.0/entity/smart-scheme/v2/mass-create?apikey={self.apikey}')

        return self.session.put(
            url=url,
            data=json.dumps(predictions, default=default),
            headers=self._headers,
        )

    def _build_url(self, endpoint: str) -> str:
        """Joins the base URL and an `endpoint` to form an absolute URL.

        Returns:
            The built URL.
                e.g. 'http://api.foodakai.com/search-api-1.0/search'
        """
        return urljoin(self.API_BASE, endpoint)

    @staticmethod
    def _get_headers() -> Dict:
        """Constructs the headers need for a request.
        Returns:
            The headers dictionary.
                e.g. {'Accept': 'application/json',
                      'Content-Type': 'application/json'}
        """
        return {'Accept': 'application/json',
                'Content-Type': 'application/json'}

    def _setup_session(self) -> requests.Session:
        """Initializes a Session and adapts several policies.

        Returns:
            requests.Session
        """
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=2,
            status_forcelist=[104, 429, 502, 503, 504],
            allowed_methods=['GET', 'POST', 'PUT'],
        )

        adapter = adapters.TimeoutHTTPAdapter(max_retries=retry_strategy, timeout=self.timeout)
        session = DataAPISession()
        session.mount('http://', adapter)

        return session
