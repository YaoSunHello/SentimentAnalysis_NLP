from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

data_path = "train.csv"
datalist = open(data_path).readlines()[1:]
#define tokenizer
tokenizer = get_tokenizer('basic_english')
# --+ deploy the tokenizer to get the vocabulary


def yield_tokens(datalist):
    for item in datalist:
        label = item[0:2]
        content = item[4:].strip()
        yield tokenizer(content)
    
vocab = build_vocab_from_iterator(yield_tokens(datalist), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

#pre-processing pipelines   

# tokenize Yelp reviews 
text_pipeline = lambda x: vocab(tokenizer(x))
# encode review labels
label_pipeline = lambda x: int(x) - 1

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

train_iter = yield_tokens(datalist)
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

for i, bathc in enumerate(dataloader):
    print(bathc)