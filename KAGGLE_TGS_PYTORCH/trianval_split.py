# -*- coding: utf-8 -*-
# @Author:Liu Chuanlong
import random
import pandas as pd


RATIO = 0.7
csv_path = "./saltdata/train.csv"


def get_ids(csv_path):
    # list_id中保存的是训练图片的名称，但是没有后缀
    df = pd.read_csv(csv_path)
    list_id = []
    for i, item in df.iterrows():
        list_id.append(item[0] + ".png")
    return list_id

def train_val_split(train_ids, ratio=0.8):

    length = len(train_ids)
    traininds = int(length * ratio)
    random.shuffle(train_ids)
    trainsections = train_ids[0:traininds]
    valinds = train_ids[traininds+1:length]
    return trainsections, valinds

def main():

    train_ids = get_ids(csv_path)
    trainsect, valsect = train_val_split(train_ids, RATIO)
    train_df = pd.DataFrame({'ids': trainsect})
    val_df = pd.DataFrame({'ids':valsect})
    train_df.to_csv("./data/train_split.csv", index=False)
    val_df.to_csv('./data/val_split.csv', index = False)


if __name__ == "__main__":
    main()
