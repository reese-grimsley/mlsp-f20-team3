import os
import gzip
import torch
import pickle
import numpy as np
import pandas as pd
from numba import cuda
import scipy.io
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, data, label_path):
        self.data = data
        self.data_len = None
        self.filename2idx = {}
        self.idx2filename = {}
        self.label = self.read_annotation(label_path)

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        filename = self.idx2filename[idx]
        x = self.data[filename]
        y = self.label[filename]
        return x, y

    def read_annotation(self, file_path):
        data = pd.read_csv(file_path, low_memory=False)
        filenames = list(data["audio_filename"])
        self.data_len = len(filenames)

        # build @self.filename2idx
        for idx in range(self.data_len):
            filename = filenames[idx]
            self.filename2idx[filename] = idx
            self.idx2filename[idx] = filename

        # build @label
        filename2label = {}
        for f in filenames:
            # -1 is just a temporary number for completing the pipeline
            filename2label[f] = -1
        return filename2label


def read_pickle(path):
    f =  open(path, 'rb')
    d = pickle.load(f)
    f.close()
    return d


def split_data(file_path="annotations.csv"):
    """
        split train/test/validation and write to csv files
    """
    data = pd.read_csv(file_path, low_memory=False)
    test = data[data["split"] == "test"]
    train = data[data["split"] == "train"]
    validate = data[data["split"] == "validate"]
    test.to_csv("test.csv", index=False)
    train.to_csv("train.csv", index=False)
    validate.to_csv("validate.csv", index=False)
    return

def convert_pickle_to_mat(path, outfile):
    d = read_pickle(path)
    d_revise = {}
    for fname in d.keys():
        fname_ = 'm' + fname.replace('.wav', "")
        print(fname_)
        d_revise[fname_] = d[fname]
    scipy.io.savemat(outfile, d_revise)


if __name__ == "__main__":
    # split_data()

    # build dataset
    logfbank_path = "logfbank_100ms.pickle"
    fbank_path = "fbank_100ms.pickle"
    mfcc_path = "mfcc_100ms.pickle"
    convert_pickle_to_mat(mfcc_path, "mfcc_100ms.mat")
    convert_pickle_to_mat(logfbank_path, "logfbank_100ms.mat")
    convert_pickle_to_mat(fbank_path, "fbank_100ms.mat")
    #convert_pickle_to_mat(fbank_path, "fbank_renamed.mat")
    # data = read_pickle(mfcc_path)
    # train_dataset = Dataset(data, "train.csv")
    #test_dataset = Dataset(data, "test.csv")
    #dev_dataset = Dataset(data, "validate.csv")
    
    # print(train_dataset.__getitem__(0))

    # # build data loader
    # train_loader_args = dict(shuffle=True, batch_size=256, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    # train_loader = DataLoader(train_dataset, **train_loader_args)

    # for i, (x, y) in enumerate(train_loader):
    #     print(x)
    #     print(y)
    #     break





