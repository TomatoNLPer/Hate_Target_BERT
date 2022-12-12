import torch
from sklearn.metrics import precision_recall_fscore_support
from optimization import BertAdam
from bert_model import senti_bert
from data_class import Sentiment, Target, Pair_Target
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import csv


def accuracy(pred, labels):
    # outputs = np.argmax(out, axis=1)
    return np.sum(pred == labels)

def macro_f1(pred, labels):
    # preds = np.argmax(y_pred, axis=-1)
    # true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(labels, pred, average='macro')
    #f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return  p_macro, r_macro,f_macro





device = torch.device('cuda:3')
batchsize= 16
num_train_epochs = 8

# dataset = Sentiment(file_name='cn_target_train.csv')
# train_sampler = RandomSampler(dataset)
# train_loader = DataLoader(dataset=dataset, sampler=train_sampler, batch_size=16)
file_path = '../data/beam_sim10000_reddit_results.json'
evalset = Pair_Target(file_name=file_path)
eval_sampler = SequentialSampler(evalset)
eval_loader = DataLoader(dataset=evalset, sampler=eval_sampler, batch_size=16)
initial_path = './bert-base-uncased/'
weight_path = r'/workspace/generate/SimCTG-main/story_generation/target_bert/best_models/conan_7_calss_acc_0.859_f1_0.7882'
weights = torch.load(weight_path)
model = senti_bert('./bert-base-uncased/')
model.load_state_dict(weights)
model.to(device)

gold_f = open('./cluster_gold_file.csv','w')
gold_w = csv.writer(gold_f,delimiter='\t')

pred_f = open('./cluster_pred_file.csv','w')
pred_w = csv.writer(pred_f,delimiter='\t')

def save_rows(writer,batch):
    for i in range(len(batch)):
        row = list(batch[i])
        writer.writerow(row)
import pandas as pd
# cc.to_csv
with torch.no_grad():
    print('start evaluation')
    score_list = []
    pred_label_list = []
    gold_label_list = []
    temp_acc = 0
    for data in tqdm(eval_loader):

        # batch_sentence = data.pop('raw_sentence')
        for item in data.keys():
            data[item] = data[item].to(device)
        gold_data = {'input_ids': data['gold_input_ids'], 'token_type_ids': data['gold_token_type_ids'],
                     'attention_mask': data['gold_attention_mask']}
        pred_data = {'input_ids': data['predict_input_ids'], 'token_type_ids': data['predict_token_type_ids'],
                     'attention_mask': data['predict_attention_mask']}
        # data = data.to(device)
        # loss = model.get_logits(data)
        # label = data['label'].detach().cpu().numpy()
        gold_logit = model.get_logits(gold_data)

        batchrows = list(gold_logit.detach().cpu().numpy())
        save_rows(gold_w,batchrows)
        # gold_labels = gold_logit.argmax(dim=1).detach().cpu().numpy()
        # gold_label_list.append(gold_labels)

        pred_logit = model.get_logits(pred_data)
        predrows = list(pred_logit.detach().cpu().numpy())
        save_rows(pred_w, predrows)
        # pred_labels = pred_logit.argmax(dim=1).detach().cpu().numpy()
        # pred_label_list.append(pred_labels)
        #
        # scores = F.softmax(pred_logit, dim=-1).detach().cpu().numpy()
        # score_list.append(scores)
        # temp_acc += accuracy()
    # print('calculate metrics')
    gold_f.close()
    pred_f.close()
    # true = np.concatenate(gold_label_list)
    # pred = np.concatenate(pred_label_list)
    # pred_list = list(pred)
    # pred_scores = np.concatenate(score_list)
    # results = []
    # # for id in range(len(pred_list)):
    # #     results.append([sentence_list[id],pred_list[id]])
    # # df1 = pd.DataFrame(data=results, columns=['sentence', 'label'])
    # # df1.to_csv('./pair_predict_target.csv')
    # acc = accuracy(pred, true) / len(pred)
    # p,r, mf1 = macro_f1(pred, true)
    # auroc = roc_auc_score(y_score=pred_scores, y_true=true, multi_class='ovo')
    # print('finish calculating')
    # print('accuracy:', acc, 'auroc:', auroc )
    # print('p:',p, 'r:',r, 'f1:', mf1)



# f = open('./file.csv','w')
# tsv_w = csv.writer(f,delimiter='\t')
#
# a = list(save_data2[2])
# tsv_w.writerow(a)
#
#
# for i in range(len(save_data2)):
#     row = list(save_data2[i])
#     tsv_w.writerow(row[0:100])

#label_map = {'JEWS':0, 'MUSLIMS':0, 'POC':1, 'MIGRANTS':2, 'WOMEN':3, 'LGBT+':4, 'DISABLED':5, 'other':6}