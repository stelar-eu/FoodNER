from typing import Iterator, List, Optional, Tuple, Union

import torch
from transformers import BertTokenizerFast as _BertTokenizer

from config.nlp_models import BERT_MODEL_NAME
from src.NLP.tokenizers.tokenizer_base import TokenizerBase


class BertTokenizer(TokenizerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = _BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        # TODO: George: train (finetune) tokenizer on our data? (tokenizer.train_new_from_iterator)
        self.bos_token: str = self.tokenizer.cls_token
        self.bos_token_id: Optional[int] = self.tokenizer.cls_token_id
        self.eos_token: str = self.tokenizer.sep_token
        self.eos_token_id: Optional[int] = self.tokenizer.sep_token_id
        self.max_length: int = self.tokenizer.model_max_length
    
    def train_new_from_iterator(self, iterator: Iterator):
        """Train a new tokenizer from an iterator.

        Args:
            iterator (Iterator): The iterator to train the tokenizer from.
        """
        self.tokenizer.train_new_from_iterator(iterator)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str], the tokenized text
        """
        return self.tokenizer.tokenize(text)
    
    def tokenize_to_ids(
        self, 
        text: str, 
        use_special_tokens: bool = True, 
        return_tensors: str = 'pt'
    ) -> Union[List[int], torch.Tensor]:
        """Tokenize a text and return the token ids.

        Args:
            text (str): The text to tokenize.
            use_special_tokens (bool): Whether to use the special bert tokens ([CLS], [SEP], [PAD]). Defaults to True.
            return_tensors (str): Whether to return a tensor or a list. None means list. Defaults to 'pt'.

        Returns:
            List[int], the token ids
        """
        if use_special_tokens:
            return self.tokenizer.encode(text, return_tensors=return_tensors)

        return self.tokenizer.convert_tokens_to_ids(self.tokenize(text))
    
    def span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        """Tokenize a text and return the start and end indices of each token.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[Tuple[int, int]]: The tokenized text.
        """
        batch_encoder = self.tokenizer(
            text,
            is_split_into_words=False,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )
        spans = []
        # hacky way to get the spans without having to extract the total number of tokens.
        for ii in range(len(text)):

            try:
                w2c = batch_encoder.word_to_chars(ii)
                spans.append((w2c.start, w2c.end))

            except TypeError:
                # there are no more words
                break

        return spans
