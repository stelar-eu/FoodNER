import requests

from loguru import logger

from src.etl.api.exceptions import DataAPIResponseError


class DataAPISession(requests.Session):
    """This is a very thin wrapper around the requests Session object which allows
    us to wrap the response handler in order to handle it in a convenient way."""

    @staticmethod
    def _handle_response(response: requests.models.Response):
        """Handles the response received from api.foodakai.com

        Args:
            response (requests.models.Response): The response object.

        Returns:
            The json-encoded content of a response, if any.
        """
        if not response.ok:

            if response.status_code == requests.codes['not_found']:
                logger.error(f'foodakai api error [{response.status_code}]: {response.text}')
                return {}

            raise DataAPIResponseError(f'foodakai api error [{response.status_code}]: {response.text}')

        try:
            json_response_content = response.json()

        except ValueError:
            raise DataAPIResponseError(f'foodakai api invalid JSON response: {response.text}')

        return json_response_content

    def request(self, *args, **kwargs):
        """Wraps Session.request and handles the response."""
        response = super(DataAPISession, self).request(*args, **kwargs)

        return self._handle_response(response)
