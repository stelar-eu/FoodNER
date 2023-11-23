from typing import List, Tuple, Iterator

from nltk import WhitespaceTokenizer

from src.NLP.tokenizers.tokenizer_base import TokenizerBase


class NLTKTokenizer(TokenizerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = WhitespaceTokenizer()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: The tokenized text.
        """
        return self.tokenizer.tokenize(text)
    
    def span_tokenize(self, text: str) -> Iterator[Tuple[int, int]]:
        """Tokenize a text and return the start and end indices of each token.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[Tuple[int, int]]: The tokenized text.
        """
        return self.tokenizer.span_tokenize(text)
