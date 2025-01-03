import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


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




def return_dict_count(data_list, top, UNK="<UNK>", PAD = "<PAD>"):
    dict = {}
    for item in data_list:
        label = item[0]
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


def load_data(data_list):
    max_len_seq = 0
    data = []
    for item in data_list:
        label = item[2:4]
        text = str(item[6:])
        token = tokenizer(text)

    if max_len_seq < len(token):
        max_len_seq = len(token)
    
    data.append([label, token])

    return data, max_len_seq

data_list = open("train_set.csv").readlines()[1:] #train_dataset
top = 5000 # top is the length of our wrd dictionary
diction = return_dict_count(data_list, top = top)


"""file = open("dictionary", "w")
for item in diction.keys():
    file.writelines("{},{}\n".format(item, diction[item]))
file.close()"""

