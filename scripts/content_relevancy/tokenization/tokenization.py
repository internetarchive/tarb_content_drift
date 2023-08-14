from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    XLNetTokenizer,
)

# Define tokenizers
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
