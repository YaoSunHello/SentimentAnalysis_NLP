from matplotlib.pyplot import text
import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from LSTMmodel import Model
#from model import Model
from load_data import text_process
import time




'''
dict_size = 5002#unique words length
embed_size = 128
num_class = 2



linear model
model = Model(dict_size,embed_size,num_class)
'''

#lstmmodel
dict_size = 5002
embed_size = 128
num_class = 2
hidden_size = 726
num_layer = 3
word_length = 2000 #the same as load_data method max_len
model = Model(dict_size,embed_size,num_class,hidden_size,num_layer,word_length)




learn_rate = 5 #for loss function
batch_size = 32
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learn_rate)
processed_text = text_process("train_set.csv")
train_batch = DataLoader(processed_text, batch_size = batch_size, shuffle = True, drop_last=True)

def train():
    model.train()
    epoches = 10
    total_acc, total_count = 0,0
    start_time = time.time()
    log_interval = 10

    for epoch in range(epoches):
        for i, batch in enumerate(train_batch):

            label,data = batch

            optimizer.zero_grad()
            pred = model.forward(data)
            loss_val = loss_func(pred, label)

        
            loss_val.backward()
            optimizer.step()

            total_acc += (pred.argmax(1)==label).sum().item()
            total_count += label.size(0)
            
            if i % log_interval and i >0:
                elapsed = time.time()-start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| accuracy {:8.3f}'.format(epoch, i, len(train_batch),
                                                total_acc/total_count))

       



####test the model below
test_text = text_process("valid_set.csv")
test_batch = DataLoader(test_text, batch_size = batch_size, shuffle = True, drop_last=True)





def test():
    total_acc, total_count = 0,0
    with torch.no_grad():
        for i, batch in enumerate(test_batch):
                label,data = batch

                pred = model.forward(data)
                #loss_val = loss_func(pred, label)

                total_acc += (pred.argmax(1)==label).sum().item()
                total_count += label.size(0)
                real_acc = total_acc/total_count
                
    print('-' * 59)
    print('valid accuracy {:8.3f} '.format(real_acc))

train()
test()
            

#the first model 
#torch.save(model.state_dict(), "LinearModel.pth")
#second model 
torch.save(model.state_dict(),"LSTMModel.pth" )
