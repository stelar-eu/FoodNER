import json
import string
from typing import List, Dict, Tuple

from tqdm import tqdm
from loguru import logger
from datasets import Dataset, DatasetDict

from config.nlp_models import DEFAULT_NER_LABEL, BERT_IGNORE_INDEX
from src.NLP.nlp_base import NLPBaseDataset
from src.NLP.tokenizers.bert import BertTokenizer
from src.tools.general_tools import get_filepath, dump_pickled_data
from src.tools.text_tools import exclude_list_duplicates, preprocess_text


class BertProcessor(NLPBaseDataset):
    def __init__(
        self, 
        annotations_filepath: str,
        *args, **kwargs
    ) -> None:
        """ Prepare data for spacy models
            Args:
                annotations_filepath (str): Path to the jsonl file that contains the data.
        """
        self.tokenizer = BertTokenizer()
        self.max_length: int = self.tokenizer.max_length
        NLPBaseDataset.__init__(self, annotations_filepath, *args, **kwargs)
    
    @property
    def vocab(self) -> List[str]:  # pragma: no cover
        return list(self.tokenizer.tokenizer.vocab.keys())
    
    @property
    def word2idx(self) -> Dict[str, int]:  # pragma: no cover
        return self.tokenizer.tokenizer.vocab
    
    @property
    def idx2word(self) -> Dict[int, str]:
        return {v: k for k, v in self.word2idx.items()}
    
    def __len__(self) -> int:
        """Property for the length of the dataset.

        Returns:
            int, the length of the dataset
        """
        return len(self.data['tokens'])
    
    def load_data(self, annotations_filepath, self_assign: bool = True) -> Dict[str, List]:  # pragma: no cover
        """Load annotated data from json file.

        Args:
            annotations_filepath (str): Path to the jsonl file that contains the annotated data.
            self_assign (bool): If True, the loaded data will be assigned to the data attribute.

        Returns:
            Dict[str, List]: The loaded data.
            {
                'tokens': List[str], 'ner_tags': List[str], 'ner_tags_encoded': List[int]
            }
        """
        def convert_single_entry(file: Dict) -> Dict[str, List[str]]:  # pragma: no cover
            """Convert a single entry to the desired format. The idea is to tokenize the input text
               and align the labels to the tokens. The difficulty lies on that each label can be a span
               of multiple tokens.

            Args:
                file (Dict): The annotated data. A single line of the original jsonl file.

            Returns:
                Tuple[List[str], List[str]]: The converted entry. The text is tokenized by using 
                    the same spans that are using for the label alignment.
            """
            spans = self.tokenizer.span_tokenize(file['data'])
            token_text: List = [file['data'][span[0]:span[1]] for span in spans]
            new_labels: List = [DEFAULT_NER_LABEL] * len(spans)

            for annotation in file['label']:
                # start = annotation[0], end = annotation[1], label = annotation[2]
                char_start, char_end = annotation[0], annotation[1]
                for ii, span in enumerate(spans):
                    token_start, token_end = span
                    if (char_start <= token_start <= char_end) and (char_start <= token_end <= char_end):
                        new_labels[ii] = annotation[2]

            if len(spans) != len(new_labels):
                raise ValueError(f'The number of spans {len(spans)} and labels {len(new_labels)} do not match.')

            return {'tokens': token_text, 'ner_tags': new_labels}

        n_ignored_entries: int = 0
        with open(annotations_filepath, 'r') as fin:
            data = {'tokens': [], 'ner_tags': [], 'original_text': []}

            for line in tqdm(fin):
                # load json line with annotations
                entry: Dict = json.loads(line)

                if len(entry['label']) == 0:  # no label in this entry
                    n_ignored_entries += 1
                    continue

                # remove unwanted symbols (e.g. leftovers from table conversion '||')
                entry = preprocess_text(entry)
                original_text = entry['data']

                # exclude duplicates
                entry['label']: List = exclude_list_duplicates(entry['label'])
                # remove spaces from labels
                entry['label'] = [tuple([label[0], label[1], label[2].replace(' ', '')]) for label in entry['label']]

                # get the data in pytorch format (tokenized)
                entry: Dict[str, List[str]] = convert_single_entry(entry)

                # convert labels to IOB format
                entry = self._ner_labels_to_iob_format(entry)

                # clean the labelled punctuation tokens
                entry = self._clean_labeled_punctuation(entry)

                # add the entry to the data
                data['tokens'].append(entry['tokens'])
                data['ner_tags'].append(entry['ner_tags'])
                data['original_text'].append(original_text)

            # make sure it's ordered before calculating label2idx and saving it
            self.labels = sorted(
                list(self.labels),
                key=lambda x: 'A' + x if x == DEFAULT_NER_LABEL else x.split('-')[1] + x.split('-')[0]
            )

            # add labels to the tokenizer's vocab
            # self.tokenizer.tokenizer.add_tokens(list(self.labels))
            label2idx = {lab: ii for ii, lab in enumerate(self.labels)}
            data['ner_tags_encoded'] = [
                [label2idx[label_token] for label_token in entities] for entities in data['ner_tags']
            ]
        
        logger.info(f"Loaded {len(data['tokens'])} entries from {annotations_filepath} "
                    f"and ignored {n_ignored_entries} entries.")

        if self_assign:
            self._data = data
        
        return data
    
    def _ner_labels_to_iob_format(self, entry: Dict[str, List[str]]) -> Dict[str, List[str]]:  # pragma: no cover
        """Convert the ner labels to the IOB format.

        Args:
            entry (Dict): The entry to convert.
                          {'tokens': List[str], 'ner_tags': List[str]}

        Returns:
            Dict: The converted entry.
                  {'tokens': List[str], 'ner_tags': List[str]}
        """
        # convert labels to IOB format
        new_labels = [DEFAULT_NER_LABEL] * len(entry['ner_tags'])

        for ii, label in enumerate(entry['ner_tags']):

            if label != DEFAULT_NER_LABEL:
                if ii == 0:
                    new_labels[ii] = f'B-{label}'

                elif entry['ner_tags'][ii-1] == label:
                    new_labels[ii] = f'I-{label}'

                else:
                    new_labels[ii] = f'B-{label}'

        self.labels.update(new_labels)
        entry['ner_tags'] = new_labels

        return entry
    
    @staticmethod
    def _clean_labeled_punctuation(entry: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Remove the labels from punctuation tokens.
           This must be called AFTER the labels are converted to IOB format.
        """
        for ii, token in enumerate(entry['tokens']):

            if token in string.punctuation:
                entry['ner_tags'][ii] = DEFAULT_NER_LABEL

        return entry

    def to_bert_format(self, base_path: str) -> DatasetDict:  # pragma: no cover
        """Convert the data to bert format where the vocab is taken from bert's vocab.

        Args:
            base_path (str): The destination folder for the dataset.
        
        Returns:
            DatasetDict: The data in hugginface's Dataset format (can be passed easily in a pytorch dataloader).
        """
        train_data = {
            'tokens': self.data['tokens'][0:self.num_train_examples],
            'ner_tags_encoded': self.data['ner_tags_encoded'][0:self.num_train_examples]
        }

        eval_data = {
            'tokens': self.data['tokens'][self.num_train_examples:],
            'ner_tags_encoded': self.data['ner_tags_encoded'][self.num_train_examples:]
        }
        eval_original_text = self.data['original_text'][self.num_train_examples:]

        data = DatasetDict({
            'train': Dataset.from_dict(train_data),
            'eval': Dataset.from_dict(eval_data),
            'all': Dataset.from_dict(
                {data: values for data, values in self.data.copy().items() if data in ['tokens', 'ner_tags_encoded']}
            )
        })

        # remove unnecessary columns
        tokenized_hf_data = data.map(self._adjust_tokens_to_labels, batched=True)
        tokenized_hf_data = tokenized_hf_data.remove_columns(['tokens', 'ner_tags_encoded'])

        # store data
        tokenized_hf_data.save_to_disk(base_path)
        dump_pickled_data(get_filepath(base_path, 'labels.pkl'), self.labels)
        dump_pickled_data(get_filepath(base_path, 'eval_original_text.pkl'), eval_original_text)

        logger.info(f"Number of train examples: {len(train_data['tokens'])}")
        logger.info(f'Number of eval examples: {len(eval_original_text)}')

        return tokenized_hf_data

    def _adjust_tokens_to_labels(self, data):  # pragma: no cover
        """Tokenize the data and adjust the labels to match the tokenization.
           This is required because the tokenizer splits the words into sub-words
           and so there not a 1-1 mapping between the labels and the tokens.
        """
        # Credit: https://huggingface.co/docs/transformers/tasks/token_classification
        tokenized_inputs = self.tokenizer.tokenizer(
            data['tokens'],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length
        )

        labels = []
        for ii, label in enumerate(data['ner_tags_encoded']):
            # map tokens to their respective word.
            word_ids = tokenized_inputs.word_ids(batch_index=ii)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100 (BERT_IGNORE_INDEX)

                if word_idx is None:
                    label_ids.append(BERT_IGNORE_INDEX)

                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])

                else:
                    label_ids.append(BERT_IGNORE_INDEX)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs['labels'] = labels

        return tokenized_inputs
