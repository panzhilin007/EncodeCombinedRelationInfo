import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class LAWDataset(Dataset):
    def __init__(self, tokenizer, config, is_train):
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        self.device = self.config.device
        if is_train:
            self.json_data = json.load(open(self.config.train_path, encoding="utf8"))
        else:
            self.json_data = json.load(open(self.config.valid_path, encoding="utf8"))
        self.rel2id = json.load(open(self.config.rel2id, encoding="utf8"))[1]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        ins_json_data = self.json_data[idx]
        text = ins_json_data["text"]
        tokens = tokenizer.tokenize(text)
        text_len = len(tokens)
        triples = ins_json_data["triple_list"]
        s2ro_map = {}
        for triple in triples:
            triple = (
                tokenizer.tokenize(triple[0]),
                triple[1],
                tokenizer.tokenize(triple[2]),
            )
            sub_head_idx = find_head_idx(tokens, triple[0])
            obj_head_idx = find_head_idx(tokens, triple[2])
            if sub_head_idx != -1 and obj_head_idx != -1:
                relation = (
                    sub_head_idx,
                    sub_head_idx + len(triple[0]) - 1,
                    obj_head_idx,
                    obj_head_idx + len(triple[2]) - 1,
                )
                rel = self.rel2id[triple[1]]
                s2ro_map[str(rel)] = []
                s2ro_map[str(rel)].append(relation)
        rels = np.zeros((text_len, 11))
        class_msk = np.ones((text_len, 11))
        rel_mask = np.zeros(self.config.rel_num)
        for ro in s2ro_map:
            for rel in s2ro_map[ro]:
                rels[rel[0]: rel[1]+1, int(ro)] = 1
                rels[rel[2]: rel[3]+1, int(ro)] = 2
                class_msk[rel[0]: rel[1]+1, int(ro)] = 2
                class_msk[rel[2]: rel[3]+1, int(ro)] = 2
                rel_mask[int(ro)] = 1
        ids = tokenizer(text)
        token_ids, segment_ids = ids["input_ids"], ids["token_type_ids"]
        masks = segment_ids
        if len(token_ids) > text_len:
            token_ids = token_ids[1:-1]
            masks = masks[:text_len]
        token_ids = np.array(token_ids)
        masks = np.array(masks) + 1
        return tokens, token_ids, masks, text_len, triples, rels, class_msk, rel_mask

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        batch.sort(key=lambda x: x[3], reverse=True)
        tokens, token_ids, masks, text_len, triples, rels, class_msk, rel_msk = zip(*batch)
        cur_batch = len(batch)
        max_len = self.config.max_len
        # max_len = max(text_len)
        batch_token_ids = torch.LongTensor(cur_batch, max_len).zero_()
        batch_masks = torch.LongTensor(cur_batch, max_len).zero_()
        batch_rels = torch.Tensor(cur_batch, max_len, 11).zero_()
        class_mask = batch_rels + 1
        rel_mask = torch.Tensor(cur_batch, self.config.rel_num).zero_()
        for i in range(cur_batch):
            batch_token_ids[i, : text_len[i]].copy_(torch.from_numpy(token_ids[i]))
            batch_masks[i, : text_len[i]].copy_(torch.from_numpy(masks[i]))
            batch_rels[i, : text_len[i], :].copy_(torch.from_numpy(rels[i]))
            class_mask[i, : text_len[i], :].copy_(torch.from_numpy(class_msk[i]))
            rel_mask[i, :].copy_(torch.from_numpy(rel_msk[i]))
        return {
            "tokens": tokens,
            "token_ids": batch_token_ids,
            "mask": batch_masks,
            "rels": batch_rels,
            "triples": triples,
            "class_mask": class_mask,
            "rel_mask": rel_mask
        }


def get_loader(config, num_workers=0, is_train=True):
    dataset = LAWDataset(tokenizer, config, is_train)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return data_loader


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
