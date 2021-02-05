import torch


class Config(object):
    def __init__(self):
        self.learning_rate = 3e-5
        self.batch_size = 16
        self.max_epoch = 400
        self.max_len = 135
        self.rel_num = 11
        self.tag_num = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_path = 'data\\train_triples.json'
        self.valid_path = 'data\\valid_triples.json'
        self.rel2id = 'data\\rel2id.json'

        self.lstm_in = 768
        self.lstm_out = 64
        self.if_bidirectional = False
        self.conv_kernel = 16
        self.pool_kernel = 2
