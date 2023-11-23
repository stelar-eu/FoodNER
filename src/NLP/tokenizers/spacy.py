from typing import List, Tuple

import spacy

from src.NLP.tokenizers.tokenizer_base import TokenizerBase


class SpacyTokenizer(TokenizerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load('en_core_web_sm')

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str], the tokenized text
        """
        doc = self.nlp(text)

        return [token.text for token in doc]
    
    def span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        """Tokenize a text and return the start and end indices of each token.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[Tuple[int, int]], the start and end indices of text tokens.
        """
        doc = self.nlp(text)

        return [(token.idx, token.idx + len(token)) for token in doc]
