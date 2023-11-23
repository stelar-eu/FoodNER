from abc import ABC, abstractmethod
from typing import List, Tuple


class TokenizerBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str], the tokenized text.
        """
        pass
    
    @abstractmethod
    def span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        """Tokenize a text and return the start and end indices of each token.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[Tuple[int, int]], the tokenized text.
        """
        pass
