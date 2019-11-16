import random
import copy
import json
import re,pprint


class ToyDataset:
    def __init__(self):
        with open('C:/Users/t-linxli/Desktop/out_rewrite.txt') as train_src:
            self.train_src_data = train_src.readlines()
        with open('C:/Users/t-linxli/Desktop/out_rewrite.txt') as train_tgt:
            self.train_tgt_data = train_tgt.readlines()
        with open('C:/Users/t-linxli/Desktop/out_rewrite.txt') as test_src:
            self.test_src_data = test_src.readlines()
        with open('C:/Users/t-linxli/Desktop/out_rewrite.txt') as test_tgt:
            self.test_tgt_data = test_tgt.readlines()

        self.batch_id = 0


    def get_batch(self, batch_size=5):
        if self.batch_id+batch_size>len(self.train_src_data):
            self.batch_id = 0
            # random.shuffle(self.train_src_data)
        batch_source = self.train_src_data[self.batch_id:self.batch_id + batch_size]
        batch_target = self.train_tgt_data[self.batch_id:self.batch_id + batch_size]
        self.batch_id = self.batch_id + batch_size
        return batch_source, batch_target

    def get_test_data(self):
        batch_source = self.test_src_data
        batch_target = self.test_tgt_data
        return batch_source, batch_target












if __name__ == '__main__':
    dataset = ToyDataset()
    pprint.pprint(dataset.get_batch())
