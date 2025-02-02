"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx.lower()
                    text += " "
                label = int(line[0]) - 1
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        #先讀取sentence，在讀取其中的每個word，如果word在self.dict，則將word轉成index，否則存為-1，一個sentence存成一個list
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1
             for word in word_tokenize(text=sentences)] for sentences in sent_tokenize(text=text)]

        #將長度不到的word padding
        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)
        # 將長度不到的sentence padding
        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        #將單一sentences list轉成sentences[max_length_word][max_length_sentences]
        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label


if __name__ == '__main__':
    test = MyDataset(data_path="../data/test.csv", dict_path="../data/glove.6B.50d.txt")
    #第0篇文章，30個sentence，35個字(30,35)
    print (test.__getitem__(index=1)[0].shape)
