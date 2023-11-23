# Named Entity Recognition Package

This is a top-level description of the data-processing, model-training, and model-prediction methods we used.

Data Preprocessing
--

We have three dataset formats (out of which only the Spacy and bert ones are used).

Regardless of the format, we always remove certain symbols that are deemed unnecessary (see config).

### Tokenization

There are many tokenization possibilities depending on what you want to achieve. For word-level tokenization it is
advised to use the Spacy tokenizer which easily provides you with the span tokenize method that returns the start and
end indices of each token (word). This is extremely useful since we want to map the start and end character ids of each
word with the corresponding ids of each label from the original `jsonl` file. A label (name entity) may actually consist
of two or more words, so a 1-1 mapping is not always feasible.

Regarding the bert format, we use huggingface's (HF's) `BertTokenizerFast` class to tokenize and encode our sentences.

### Datasets


The Spacy datasets are produced by getting the spans of each labelled word and passing them to a `DocBin` object.
This object can then easily be saved and read by the model we want to train (see the `.Spacy` files produced
under the `results/dataset` directory).

The bert dataset is a bit more complex is the tokenization does not occur on a word-level. For example, the
word `testing` would be tokenized to `test`, and `##ing`. This means that we have to align the label start and end
character ids to match the new tokens. This happens in `_adjust_tokens_to_labels` and a similar implementation can be
found in the "official" [HF NER tutorial](https://huggingface.co/docs/transformers/tasks/token_classification). 

For bert, we also convert the labels to IOB format. This means that each token that corresponds to the start of a label
will get the `B-` prefix and the corresponding tokens will get the `I-` prefix. In addition, all in-between punctuation
symbols are not considered as valid NER labels (e.g. in `Cow, Sheep [ANIMAL]`, the comma will not be assigned
the `[Animal]` label).

Model Training
--

### Spacy model

Spacy models are separated into two categories. First, we have the transition-based models that use more "traditional"
tools to train NER systems. On top of that, we also used transformer models which were introduced in the 3rd version
of Spacy.

- Transition-based models:
  - Type: neural networks
  - Pre-tokenization: We used the pre-trained Spacy en_core_web_lg model (also transition based).
    - trained on written web text (blogs, news, comments) so it doesn't have any prior knowledge on food technology
      terminology topics.
  - Our NER model:
    - Tok2Vec: Extract dynamic representations of tokens to match the current context. It can work alongside static
      vectors such as FastText embeddings. In our case we train the tok2vec from scratch (it is part of the pipeline).
    - Transition model for NER: (https://arxiv.org/pdf/1603.01360.pdf)
        - A model that constructs and labels pieces of input sentences using a transition-based parsing algorithm with
          states represented by stack LSTMs
        - Directly constructs representations of multi-token names.
        - Incrementaly constructs chunks of the input by using a stacked data structure.
        - Stack-LSTMs (LSTMs augmented by an additional stack pointer): Mantain a "summary embedding" of its contents
        - Then use Spacy's `Embed, Encode, Attend, Predict` Framework:
            - **Embed**: get representation of each word (token)
            - **Encode**: Summary vector for each sentence (sentence embed) (trigram CNN instead of LSTM)
            - **Attend**: apply attention (self)
            - **Predict**: sort of like beam search. first uses an MLP to get the sequence predictions and then
                           performs a search.
        - Transition models are good for sequence tagging (since they are primarily designed for that task) and they
          can define arbitrary sequences quite easily.
    
- Transformer Models:
    - Come with the Spacy-transformers package that uses a pretrained transofmer and tunes it on our data.
    - In our case we are using the `bert-base-cased` model (taken from hugginface's repositories) and finetune it for
      the task of NER (token classification).
    - For the final predictions, we use the same step as with the transition-based model above. I.e., we use an MLP to
      get the sequence predictions and then perform a search.
    - The encodings of the transformer are also fine-tuned, though.
    

### HF Pytorch Model:

We used HugginFace's interface to fine-tune a pretrained `TokenClassification` model (which is how they refer to
Named Entity Recognition). So it is not per se a sequence to sequence classification task, but rather a token
classification task where the input and predicted output always have the same length.

In particular, we used the `bert-base-cased` model as the pre-trained encoder. On top of it, the `TokenClassification`
class also adds a token classification head which is just a linear layer (MLP) on top of the hidden-states output.
For each token we get the logits (kind-of like probabilities) to belong to each class.

Regarding the preprocessing, it is important that we use the exact same tokenizer model as the one used to initialize
the `TokenClassification` instance (`bert-base-cased` in our case).

The issue with this approach is that the tokenization does not happen on a word-level. This means that instead
of 1 prediction, we now need to make 2 predictions. While this is not an issue with training, it could result to issues
when predicting since the output predictions will correspond to tokens and not words. **NOTE**: This means that you will
have to find a way to convert the token-level predicted labels of this model to word-level ones.

At last, **note** that the model is trained on IOB-formatted data (each label has the `B-` and `I-` labels as a prefix).
This increases the labels from 18+1 (the default label) to 36+1. That, along with the fact that we don't have a lot of
data, could result in an underperforming model.


Evaluation
--

  - We extract stats for four eval categories.
    - Type: we want at least some overlap between our predicted tag and the golden truth, AND the correct type. E.g. if
      the ground truth is "warfarin [DRUG]" and the prediction is "of warfarin [DRUG]" then this is correct.
    - Partial: We don't care about the predicted entity type as long as there is a match between tehe predicted
      boundaries and the ground truth boundaries. E.g. GT "propranolol [DRUG]" and prediction="propranolol [BRAND]" then
      this is still considered correct. If GT="warfarin [DRUG]" and pred="of warfarin [DRUG]" then this is partially correct.
    - Exact: We don't cate abut the rpedicted entity type as long as there is an EXACT match between the predicted
      and GT boundaries.
    - Strict: there should be a 1:1 matching between the predicted and GT labels.
  - In our case, **we are primarily interested in the "Type" and "Strict"** metrics.

*Note: The evaluation of the HF-BERT model occurs at every epoch when using the evaluation command. Although, there is still a separate `evaluate` command which uses the HF trained to evaluate on a provided evaluation set.*

## Future Steps:

- Spacy:
    - Pretrain tok2vec on a large set of related text and then use it to train the ner model. In the current setup the
      tok2vec model is trained along the ner model, but there may not be enough data to train a sufficient tok2vec model.
    - For the transformer model, we should also train with `roberta-base-cased` which is the model recommended by Spacy.
    - Further parameter-tuning. It would be good to use a hyper-parameter tuning library to find the best parameters for
      both the Spacy and bert HF models.