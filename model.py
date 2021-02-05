import torch
import json
import config
import data_loader
import time
from torch import nn as nn
from transformers import BertModel, BertLayer, BertConfig
from torch.autograd import Variable

# torch.cuda.set_device(1)


class LSTMS(nn.Module):
    def __init__(self, config) -> None:
        super(LSTMS, self).__init__()
        self.config = config
        self.n = self.config.rel_num
        self.device = self.config.device
        self.lstm_in = self.config.lstm_in
        self.lstm_out = self.config.lstm_out
        self.lstms = nn.ModuleList([
            nn.LSTM(self.lstm_in,
                    self.lstm_out,
                    batch_first=True,
                    bidirectional=self.config.if_bidirectional)
            for _ in range(self.n)
        ])

    def forward(self, encode):
        out = torch.zeros(encode.size(0), 135, self.n,
                          self.config.lstm_out).to(self.device)
        for i in range(self.n):
            out[:, :, i, :], (h, c) = self.lstms[i](encode)
        return out


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.bert_dim = 768
        self.rel_num = self.config.rel_num
        self.max_len = self.config.max_len
        self.device = self.config.device
        self.lr = self.config.learning_rate
        self.id2rel = json.load(open(self.config.rel2id, encoding="utf8"))[0]

        self.bert_encoder = BertModel.from_pretrained("bert-base-chinese")
        self.conv = nn.Conv1d(in_channels=self.bert_dim,
                              out_channels=self.rel_num,
                              kernel_size=self.config.conv_kernel)
        self.pool = nn.MaxPool1d(self.config.pool_kernel)
        self.lstm = nn.LSTM(input_size=self.config.lstm_in,
                            hidden_size=self.config.lstm_out,
                            batch_first=True,
                            bidirectional=self.config.if_bidirectional)
        self.lstms = LSTMS(self.config)
        # self.w = nn.Linear(in_features=self.max_len -
        #                    self.config.conv_kernel + 1, out_features=128)
        self.w = nn.Linear(in_features=self.max_len - self.config.conv_kernel +
                           1,
                           out_features=384)
        self.linears = nn.Linear(in_features=self.config.lstm_out,
                                 out_features=self.config.tag_num)
        self.rel2tag = nn.Linear(in_features=self.max_len -
                                 self.config.conv_kernel + 1,
                                 out_features=1)
        self.softmax = nn.Softmax(-1)
        self.layernorm = nn.LayerNorm(
            [self.config.max_len, self.config.lstm_in])

        self.matrix = Variable(torch.randn(128, self.bert_dim),
                               requires_grad=True).to(self.device)
        self.bertlayer1 = BertLayer(BertConfig(vocab_size=21128))
        self.bertlayer2 = BertLayer(BertConfig(vocab_size=21128))

    def criterion(self, pred, gold, mask, class_mask):
        loss = nn.CrossEntropyLoss(reduction='none')
        # loss = nn.MSELoss(reduction='none')
        los = loss(pred, gold)
        if los.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        los = los * class_mask
        los = torch.sum(los * mask) / torch.sum(mask)
        return los

    def attn(self, rel, encode):
        rel = self.w(rel)
        # similarity = torch.cosine_similarity(torch.cat(
        #     [rel.unsqueeze(1)]*self.max_len, 1), torch.max_pool1d(encode, kernel_size=6), -1)
        dotProduct = torch.bmm(torch.max_pool1d(encode, kernel_size=6),
                               rel.unsqueeze(-1))
        # matrixMap = torch.matmul(torch.matmul(torch.unsqueeze(
        #     rel, 1), self.matrix), encode.permute(0, 2, 1))
        # prob = torch.softmax(matrixMap, -1)
        # weight = torch.cat([matrixMap]*self.bert_dim, -2).permute(0, 2, 1)
        # weight = torch.cat([torch.unsqueeze(similarity, -1)]*self.bert_dim, -1)
        weight = torch.cat([dotProduct] * self.bert_dim, -1)
        context = weight * encode
        return torch.sigmoid(self.layernorm(context))

    def forward(self, data, is_train=True):
        encode = self.bert_encoder(data["token_ids"], data["mask"])[0]
        # msk = self.bert_encoder.get_extended_attention_mask(
        #     attention_mask=data["mask"], input_shape=[16, 135], device=self.config.device)
        # rel_encode = self.bertlayer1(encode, attention_mask=msk)[0]
        # text_encode = self.bertlayer2(encode, attention_mask=msk)[0]
        conv = self.conv(encode.permute(0, 2, 1))
        # conv = self.pool(conv)
        out = torch.zeros(conv.size(0), self.max_len, self.rel_num,
                          self.config.lstm_out).to(self.device)
        rel = torch.cat([torch.unsqueeze(self.w(conv), 1)] * self.max_len, 1)
        for i in range(self.rel_num):
            # context = self.attn(torch.relu(conv[:, i, :]), text_encode)
            context = torch.cat((torch.max_pool1d(encode, 2), rel[:, :, i, :]), -1)
            # context = encode
            out[:, :, i, :], (h, c) = self.lstm(context)
        # out = self.lstms(encode)
        out = self.linears(out)
        if is_train:
            rel2tag = self.rel2tag(conv)
            return out, torch.sigmoid(torch.squeeze(rel2tag, -1))
        return out

    def train(self):
        # for param in self.bert_encoder.parameters():
        #     param.requires_grad = False
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                             self.parameters()),
                                      lr=self.lr)
        # criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        bcelosss = nn.BCELoss()
        train_loader = data_loader.get_loader(conf, is_train=True)
        valid_loader = data_loader.get_loader(conf, is_train=False)
        best_f1 = (0, 0, 0)
        state = self.state_dict()
        for epoch in range(self.config.max_epoch):
            start_time = time.time()
            cur_loss1 = cur_loss2 = cur_loss = 0
            step = 0
            train_fetcher = data_loader.DataPreFetcher(train_loader)
            train_batch = train_fetcher.next()
            while train_batch is not None:
                pred, rel = self.forward(train_batch)
                # loss = criterion(pred.permute(0, 3, 1, 2), train_batch['rels'].long())
                loss1 = self.criterion(pred.permute(0, 3, 1, 2), train_batch['rels'].long(
                ), train_batch['mask'], train_batch['class_mask'])
                loss2 = bcelosss(rel, train_batch['rel_mask'])
                loss = loss1 + loss2
                # pred = self.forward(train_batch)
                loss = self.criterion(pred.permute(0, 3, 1, 2),
                                      train_batch['rels'].long(),
                                      train_batch['mask'],
                                      train_batch['class_mask'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cur_loss1 += loss1
                cur_loss2 += loss2
                cur_loss += loss
                step += 1
                train_batch = train_fetcher.next()

            print(time.strftime("%m-%d %H:%M:%S", time.localtime()), end='')
            print('  epoch: {}, spend time: {:.3f}s'.format(
                epoch + 1,
                time.time() - start_time),
                end='')
            print(', loss1:{:.4f}, loss2:{:.4f}, loss:{:.4f}'.format(
                cur_loss1/step, cur_loss2/step, cur_loss/step))
            # print(', loss:{:.4f}'.format(cur_loss/step))

            if (epoch + 1) % 20 == 0 and (epoch + 1) >= 100:
                p, r, f1 = self.eval(valid_loader)
                if f1 > best_f1[2]:
                    best_f1 = (p, r, f1)
                    state = self.state_dict()
                print(
                    'epoch {}, f1: {:.3f}, precision: {:.3f}, recall: {:.3f}'.
                    format(epoch + 1, best_f1[2], best_f1[0], best_f1[1]))
        return state

    def eval(self, valid_loader):
        correct_num, pred_num, gold_num = 0, 0, 0
        valid_fetcher = data_loader.DataPreFetcher(valid_loader)
        valid_batch = valid_fetcher.next()
        while valid_batch is not None:
            with torch.no_grad():
                pred = self.softmax(self.forward(valid_batch, is_train=False))
                for i in range(pred.size(0)):
                    gold_num += len(valid_batch['triples'][i])
                    pred_triples = self.rec(torch.round(pred[i]),
                                            valid_batch['tokens'][i])
                    pred_num += len(pred_triples)
                    for triple in pred_triples:
                        if triple in valid_batch['triples'][i]:
                            correct_num += 1
            valid_batch = valid_fetcher.next()
        p = correct_num / (pred_num + 1e-10)
        r = correct_num / (gold_num + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        return p, r, f1

    def rec(self, label, tokens):
        triples = []
        length = len(tokens)
        for rel in range(11):
            start = -1
            entities = []
            for i in range(length):
                if start == -1:
                    if get_index(label, i, rel) > 0:
                        start = i
                else:
                    if get_index(label, i, rel) == 0:
                        entities.append((''.join(tokens[start:i]),
                                         int(get_index(label, start, rel))))
                        start = -1
                    elif get_index(label, i, rel) != get_index(
                            label, start, rel):
                        entities.append((''.join(tokens[start:i]),
                                         int(get_index(label, start, rel))))
                        start = i
            triples.extend(self.relation_rec(entities, rel))
        return triples

    def relation_rec(self, entities, rel):
        front = [entity[0] for entity in entities if entity[1] == 1]
        behind = [entity[0] for entity in entities if entity[1] == 2]
        if len(front) == 0 or len(behind) == 0:
            return []
        else:
            if len(front) == 1:
                return [[front[0], self.id2rel[str(rel)], entity]
                        for entity in behind]
            elif len(front) == len(behind):
                return [[front[i], self.id2rel[str(rel)], behind[i]]
                        for i in range(len(front))]
            elif len(behind) == 1:
                return [[entity, self.id2rel[str(rel)], behind[0]]
                        for entity in front]
            else:
                return []


def get_index(label, i, k):
    return list(label[i][k]).index(max(list(label[i][k])))


conf = config.Config()

net = Net(conf).to(conf.device)
state = net.train()
torch.save(state, 'models\\model.pt')
