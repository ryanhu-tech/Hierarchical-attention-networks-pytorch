"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import sys
import csv
#原本會產生OverflowError: Python int too large to convert to C long錯誤，改採下列作法
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def matrix_mul(input, weight, bias=False):
    # input [word_length, bs, 2*hidden_size]
    feature_list = []
    #feature [bs, 2*hidden_size]
    for feature in input:
        #torch.mm是矩陣乘法 ，[bs, 2*hidden_size] x [2*hidden_size, 2*hidden_size] = [bs, 2*hidden_size]
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            #bias從[1, 2*2*hidden_size] -> [bs, 2*hidden_size]
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        #增加一個維度 [1, bs, (2*hidden_size)]
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    # [word_length, bs, (2*hidden_size)]
    # squeeze()是針對與context計算attention後最後一個維度只有一個值，用squeeze後會變[word_length, bs]
    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    #input2是對第i個字的attention score，所以要增加維度到(2*hidden_state)
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        #點對點的乘法
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    #[word_length, bs, 2 * hidden_size]
    output = torch.cat(feature_list, 0)
    #[word_length, bs, 2 * hidden_size]->[bs, 2 * hidden_size]->[1,bs, 2 * hidden_size]
    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        #將list依word或sentence長度進行排序
        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)
    #取排序80%的長度為最大長度
    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print(word)
    print(sent)






