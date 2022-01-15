"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttNet, self).__init__()
        #filepath_or_bufferg是文件路徑，quoting是設定保留的"符號數量，.values表示回傳numpy表示的值，不包含axeslabels
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        #定義<UNK>的size
        unknown_word = np.zeros((1, embed_size))
        #結合<UNK>至第0個位置
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
        #查一下nn.Parameter
        #對應公式(5)的W,b
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        #對應公式(6)的Uw
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        #word embedding
        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        #[word_length, batch_size, (2*hidden_size)]
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    #預設word_weight 及 context_weight 的分布
    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        # input [word_length, bs]
        # output [word_length, embed_size, hidden_size]
        output = self.lookup(input)
        # [word_length, bs, 2*hidden_size]
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        #自訂義的矩陣乘法
        # [word_length, bs, (2*hidden_size)]
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        #為何要交換維度?
        #[word_length, bs, (2*hidden_size)] * [ (2*hidden_size), 1] = [word_length, bs]
        #permute -> [ bs, word_length]
        output = matrix_mul(output, self.context_weight).permute(1,0)
        #[ bs, word_length] 要確定softmax做的維度及輸出維度
        output = F.softmax(output, dim=1)
        # [word_length, bs, 2*hidden_size] * [ word_length, bs] -> [1,bs, 2 * hidden_size]
        output = element_wise_mul(f_output,output.permute(1,0))

        return output, h_output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
