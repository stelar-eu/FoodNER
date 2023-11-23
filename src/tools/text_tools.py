import re
from typing import List, Dict

from config.nlp_models import SYMBOLS_TO_REMOVE


def exclude_list_duplicates(data: List) -> List:
    """Exclude duplicate items from a list.

    Args:
        data (List): The given list.

    Returns:
        List
    """
    return sorted([*set([tuple(item) for item in data])])


def ids_to_tokens(idx2word: Dict[int, str], ids: List[int]) -> List[str]:
    """Convert a list of ids to a sentence (list of tokens).

    Args:
        idx2word (Dict[int, str]): The dict with the ids (key) and tokens (value).
        ids (List[int]): A list of ids.

    Returns:
        List[str], the sentence as a list of tokens
    """
    return [idx2word[idx] for idx in ids]


def tokens_to_ids(word2idx: Dict[str, int], tokens: List[str]) -> List[int]:
    """Convert a sentence to a list of ids.

    Args:
        word2idx (Dict[str, int]): The dict with the tokens (key) and ids (value).
        tokens (List[str]): A list of tokens.

    Returns:
        List[int], a list of ids
    """
    return [word2idx[word] for word in tokens]


def extract_text_for_predict(texts: List[str]) -> List[str]:
    """Extract text from raw data.

    Args:
        texts (List[str]): The raw text data.

    Returns:
        List[str]: The extracted text (tokenized and cleaned).
    """
    final_text = []

    for text in texts:

        for symbol in SYMBOLS_TO_REMOVE:
            text = re.sub(symbol, ' ', text)

        text = re.sub(r'\s+', ' ', text)
        final_text.append(text)

    return final_text


def preprocess_text(entry: Dict) -> Dict[str, str]:
    """Preprocess the text. The difficulty here lies on that we have to align
       the label start and end characters with the processed text.

    Args:
        entry (Dict): An entry text (json line) from the annotated json file.

    Returns:
        Dict: The preprocessed en try.
    """
    text: str = entry['data']
    labels: List = sorted(entry['label'])
    for symbol, symbol_length in SYMBOLS_TO_REMOVE.items():
        # remove the symbol from the text and add a space in its place
        length_to_remove = symbol_length - 1
        matches = re.finditer(symbol, text)

        # update a copy of the labels since the matches were extracted from the original text
        tmp_labels = labels.copy()

        for match in matches:
            _, end = match.span()

            for ii in range(len(labels)):
                label_start, _, _ = labels[ii]

                # if the label start is after the end of the match, we need to update the label
                # start and end of the current label and all the following labels
                if label_start >= end:
                    for jj in range(ii, len(labels)):
                        label_start, label_end, label_type = tmp_labels[jj]
                        label_start -= length_to_remove
                        label_end -= length_to_remove
                        tmp_labels[jj] = (label_start, label_end, label_type)

                    break

        # update the labels and the text (by removing the symbol)
        labels = tmp_labels
        text = re.sub(symbol, ' ', text)

    return {'data': text, 'label': labels}
