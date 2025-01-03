import torch 
from  torch.utils.data import DataLoader, Dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import numpy as np

def tokenizer(text):
    tk = TreebankWordTokenizer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    token = tk.tokenize(review)
    return token


def load_data(data_list):
    max_len_seq = 2000
    data = []
    for item in data_list:
        label = item[0]
        text = str(item[4:])
        token = tokenizer(text)

        if max_len_seq < len(token):
            max_len_seq = len(token)
    
        data.append([label, token])

    return data, max_len_seq



def return_dict_count(data_list, top=5000, UNK="<UNK>", PAD = "<PAD>"):
    dict = {}
    for item in data_list:
        text = str(item[4:])
        token = tokenizer(text)

        for t in token:
            if t in dict.keys():
                dict[t] += 1
            else:
                dict[t] = 1
    dict = sorted([_ for _ in dict.items() if _[1] > 1], key = lambda x:x[1],reverse = True)[:top]
    dict = {word_count[0]: idx for idx, word_count in enumerate(dict)}
    dict.update({UNK:len(dict), PAD:len(dict)+1})
    return dict


#define a class to enter the elements
class text_process(Dataset):
    def __init__(self, file_path):
        super(text_process, self).__init__()
        self.path = file_path
        self.data_list = open(self.path).readlines()[1:]
        self.data, self.max_len_seq = load_data(self.data_list)
        self.dict = return_dict_count(self.data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = data[0]
        text = data[1]
        input_idx = []
        for word in text:
            if word in self.dict.keys():
                input_idx.append(self.dict[word])
            else:
                input_idx.append(self.dict["<UNK>"])
        if len(text) < self.max_len_seq:
            input_idx += [self.dict["<PAD>"] for _ in range(self.max_len_seq - len(input_idx))]

        #label = torch.tensor(label,dtype= torch.int32)
        input_idx = torch.tensor(input_idx)
        label = torch.tensor(int(label),dtype=torch.long)

        return label, input_idx




'''
DL = text_process("train_set.csv")

DL = DataLoader(DL, batch_size = 128, shuffle=True,drop_last= True)
'''