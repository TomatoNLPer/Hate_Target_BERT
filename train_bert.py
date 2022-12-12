import torch
from sklearn.metrics import precision_recall_fscore_support
from optimization import BertAdam
from bert_model import senti_bert
from data_class import Sentiment
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm

def accuracy(pred, labels):
    # outputs = np.argmax(out, axis=1)
    return np.sum(pred == labels)

def macro_f1(pred, labels):
    # preds = np.argmax(y_pred, axis=-1)
    # true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(labels, pred, average='macro')
    #f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return f_macro





device = torch.device('cuda:2')
batchsize= 16
num_train_epochs = 8

dataset = Sentiment(file_name='new_cn_target_train.csv')
train_sampler = RandomSampler(dataset)
train_loader = DataLoader(dataset=dataset, sampler=train_sampler, batch_size=16)

evalset = Sentiment(file_name='new_cn_target_dev.csv')
eval_sampler =SequentialSampler(evalset)
eval_loader = DataLoader(dataset=evalset, sampler=eval_sampler, batch_size=16)
weight_path = './bert-base-uncased/'
model = senti_bert(weight_path).to(device)
num_train_steps = int(
            len(dataset) / batchsize  * num_train_epochs)
optimizer = BertAdam(model.parameters(),
                     lr=4e-5,
                     warmup=0.1,
                     t_total=num_train_steps)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[3,6],gamma=0.9)
max_acc = 0
for i in range(num_train_epochs):

    with torch.no_grad():
        print('start evaluation')
        true_label_list = []
        pred_label_list = []
        temp_acc = 0
        for data in tqdm(eval_loader):
            for item in data.keys():
                data[item] = data[item].to(device)
            # data = data.to(device)
            loss = model.get_logits(data)
            label = data['label'].detach().cpu().numpy()
            logit = model.get_logits(data)
            pred_labels = logit.argmax(dim=1).detach().cpu().numpy()
            true_label_list.append(label)
            pred_label_list.append(pred_labels)
            # temp_acc += accuracy()
        # print('calculate metrics')
        true = np.concatenate(true_label_list)
        pred = np.concatenate(pred_label_list)
        acc = accuracy(pred, true) / len(pred)
        mf1 = macro_f1(pred, true)
        # print('finish calculating')
        print('accuracy:', acc, 'f1:', mf1)

        if acc > max_acc:
            path = './best_models/' + '7_calss_' + 'acc_'+str(acc)[:6]+'_f1_'+str(mf1)[:6]
            torch.save(model.state_dict(), path)
            max_acc = acc
    for data in tqdm(train_loader):

        for item in data.keys():
            data[item] = data[item].to(device)
        # data = data.to(device)
        loss = model(data)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        print(loss)

