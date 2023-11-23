from src.etl.api.interface import APIInterface


class Extractor:
    """The general Extractor."""

    def __init__(self, data_from: str, data_until: str, lang: str) -> None:
        """Initializes an extractor class.

        Args:
            data_from (str): Starting date of the retrieved data from DataAPI. Input date format YYYY-MM-DD.
            data_until (str): Last day of the retrieved data from DataAPI. Input date format YYYY-MM-DD.
            lang (str): The selected language of the documents.
        """
        self.data_from: str = data_from
        self.data_until: str = data_until
        self.lang: str = lang
        self.api = APIInterface()
