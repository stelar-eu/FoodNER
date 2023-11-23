from typing import List

from src.etl.extraction import Extractor
from src.tools.general_tools import dump_json_data


class TextExtractor(Extractor):

    def __init__(self, data_from: str, data_until: str, lang: str) -> None:
        """Data preprocessor extracts the final dataset.

        Args:
            data_from (str): Starting date of the retrieved data from DataAPI. Input date format YYYY-MM-DD.
            data_until (str): Last day of the retrieved data from DataAPI. Input date format YYYY-MM-DD.
            lang (str): The selected language of the documents.
        """
        super().__init__(data_from, data_until, lang)

    def corpus(self, filepath: str) -> None:
        """Extract product incidents and data preparation.

        Args:
            filepath (str): The destination path of the jsonl file.

        Returns:
            None
        """
        # extract corpus
        corpus: List = self.api.generate_text_documents(self.data_from, self.data_until, lang=self.lang)

        # parse corpus
        # TODO: 1) per sentence or per paragraph
        #       2) use || for tables
        #       3) When we use \n

        # store corpus in a jsonl file
        dump_json_data(filepath, corpus, indent=0, lines=True)
