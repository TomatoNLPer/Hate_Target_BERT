import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class senti_bert(nn.Module):
    def __init__(self,model_path):
        super(senti_bert, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)

        self.classifier = nn.Linear(768,7)
        self.loss_function = nn.CrossEntropyLoss()
    def forward(self, batch_data):
        output = self.bert(input_ids=batch_data['input_ids'], token_type_ids=batch_data['token_type_ids'],
                           attention_mask=batch_data['attention_mask'])
        sentence_rep = output.last_hidden_state[:,0,:]
        logits = self.classifier(sentence_rep)
        loss = self.loss_function(logits,torch.tensor(batch_data['label']))
        return loss
    def get_logits(self, batch_data):
        output = self.bert(input_ids=batch_data['input_ids'], token_type_ids=batch_data['token_type_ids'],
                           attention_mask=batch_data['attention_mask'])
        sentence_rep = output.last_hidden_state[:, 0, :]
        logits = self.classifier(sentence_rep)
        return logits