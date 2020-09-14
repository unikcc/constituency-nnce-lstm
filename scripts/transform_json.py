import os
import pickle as pkl
import json
from collections import Counter
import random

random.seed(0)



class Preprocessor:

    def __init__(self):
        basename = os.path.basename(os.getcwd())
        config = json.load(open('config.json'.format(basename), 'r'))
        for cfg in config:
            self.__setattr__(cfg, config[cfg])

        self.target_dir = self.trans_json_target_dir.format(basename)
        self.data_dir = self.trans_json_data_dir
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

    def read_file(self, fileaname):
        a = open(fileaname, 'r').read().splitlines()
        text = [json.loads(w) for w in a]
        return text

    def save_file(self, data, mode='train'):
        filename = os.path.join(self.target_dir, '{}.json'.format(mode))
        outf = open(filename, 'w', encoding='utf-8')
        for i, line in enumerate(data):
            json.dump(line, outf, ensure_ascii=False)
            if i != len(data) - 1:
                outf.write('\n')
        outf.close()

    def manage(self):
        data, labels = [], []
        files = os.listdir(self.data_dir)
        for filename in files:
            filename = os.path.join(self.data_dir, filename)
            document = self.read_file(filename)
            data.append(document)
        random.shuffle(data)
        test_num, valid_num = 200, 200
        test_data, valid_data, train_data = \
            data[:test_num], \
            data[test_num:test_num + valid_num], \
            data[test_num + valid_num:]

        self.save_file(test_data, 'test')
        self.save_file(valid_data, 'valid')
        self.save_file(train_data, 'train')



if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.manage()
