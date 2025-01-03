from sklearn.model_selection import train_test_split
import pandas as pd

class Train_Test():
    def __init__(self):
        self.path = "train.csv"
        self.data = pd.read_csv(self.path)[:1000]#here select 1000 data for a fast exhibition

    def data_split(self, test_size, random_state):
        train_set, test_set = train_test_split(self.data, test_size = test_size, random_state = random_state)
        train_set = train_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)
        train_set.to_csv("train_set.csv",index=False)
        test_set.to_csv("valid_set.csv",index=False)
        return None


spliter = Train_Test()
spliter.data_split(0.3, 100)


