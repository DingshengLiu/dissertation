import random, numpy as np, pandas as pd, torch
from tool import preprocess
from torch.utils.data import Dataset, DataLoader
class TrainDataset(Dataset):
    """
    每条交互 = 一行样本
    返回:  user_id, pos_item, neg_items(K,)
    """

    def __init__(self, df, user_col, item_col, num_items,
                 k_neg=1, neg_sampling='pop', alpha=0.75):

        super().__init__()
        self.users = df[user_col].to_numpy(dtype=np.int32)
        self.items = df[item_col].to_numpy(dtype=np.int32)
        self.num_items = num_items
        self.k_neg = k_neg

        # 1) 预构 user→正集合
        self.user_pos = (
            df.groupby(user_col)[item_col]
            .apply(lambda x: set(x.tolist()))
            .to_dict()
        )

        # 2) 负采样分布
        if neg_sampling == 'pop':  # 按 item 热度的 α 次幂
            pop = df[item_col].value_counts().sort_index()
            self.item_pool = pop.index.to_numpy()
            prob = (pop ** alpha) / (pop ** alpha).sum()
            self.neg_dist = np.asarray(prob, dtype=np.float64)
        else:  # uniform
            self.neg_dist = None

    def __len__(self):
        return len(self.users)

    def _sample_neg(self, u):
        """采 K 个对 u 不在正集合里的负样本"""
        negs = []
        pos_set = self.user_pos[u]
        while len(negs) < self.k_neg:
            # 先一次抽一大批候选，再过滤，加速
            need = self.k_neg - len(negs)
            pool = np.random.choice(
                self.item_pool,  # a: 只有出现过的 id
                size=need * 3,
                p=self.neg_dist  # p: 同长度概率向量
            )
            for n in pool:
                if n not in pos_set:
                    negs.append(n)
                    if len(negs) == self.k_neg:
                        break
        return negs

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.items[idx]
        neg = self._sample_neg(u)
        return (
            torch.as_tensor(u, dtype=torch.long),
            torch.as_tensor(pos, dtype=torch.long),
            torch.as_tensor(neg, dtype=torch.long)  # shape (K,)
        )


# ---------- 用法 ----------
def build_train_loader(train_df, num_items, user_col, item_col,
                  batch_size=1024,
                 k_neg=1, num_workers=1):

    ds = TrainDataset(train_df, user_col, item_col,
                            num_items, k_neg=k_neg)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=True,  # epoch 内无放回洗牌
                      num_workers=num_workers,
                      drop_last=True,
                      worker_init_fn = preprocess.set_seed(42))


class TestDataset(Dataset):
    """
    每条交互 = 一行样本
    返回:  user_id, pos_item, neg_items(K,)
    """

    def __init__(self, df, user_col, item_col, num_items):

        super().__init__()
        self.users = df[user_col].to_numpy(dtype=np.int32)
        self.items = df[item_col].to_numpy(dtype=np.int32)
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.items[idx]
        return (
            torch.as_tensor(u, dtype=torch.long),
            torch.as_tensor(pos, dtype=torch.long),
        )


# ---------- 用法 ----------
def build_test_loader(test_df, num_items, user_col, item_col,
                  batch_size=1024,
                 k_neg=1, num_workers=1):

    ds = TestDataset(test_df, user_col, item_col,
                            num_items)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=True,  # epoch 内无放回洗牌
                      num_workers=num_workers,
                      drop_last=False)

class InBatchTrainDataset(Dataset):
    def __init__(self, df, user_col, item_col):
        # 筛出所有 (user_id, item_id) 正交互对
        self.user_ids = df[user_col].values.astype('int64')
        self.item_ids = df[item_col].values.astype('int64')

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx]

def build_train_loader_inbatch(train_df,user_col,item_col,batch_size=1024, shuffle=True, num_workers=2):
    dataset = InBatchTrainDataset(train_df,user_col,item_col)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True,
                      drop_last=True)  # drop_last 保证每个 batch 大小一致，避免 CrossEntropy 报错

class SeqDataset(Dataset):
    """
    每条样本: (hist_seq, pos_item )
    hist_seq 已左侧 padding 到长度 max_len
    """
    def __init__(self, df, max_len, pad_idx, user_id, item_id):
        super().__init__()
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.inputs, self.targets = [], []
        for _, user_hist in df.groupby(user_id):
            seq = user_hist[item_id].tolist()
            for i in range(1, len(seq)):
                hist = seq[max(0, i - max_len): i]
                hist = [pad_idx] * (max_len - len(hist)) + hist
                self.inputs.append(hist)
                self.targets.append(seq[i])  # 正样本

        self.inputs  = np.asarray(self.inputs,  dtype=np.int64)
        self.targets = np.asarray(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        hist = self.inputs[idx]
        pos  = self.targets[idx]
        return (
            torch.tensor(hist, dtype=torch.long),    # (T,)
            torch.tensor(pos,  dtype=torch.long),    # ()
        )
def build_seq_loader(df, max_len, pad_idx, user_id, item_id ,batch_size=1024, shuffle=True, num_workers=2):
    dataset = SeqDataset(df, max_len, pad_idx, user_id, item_id)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True,
                      drop_last=True,
                      worker_init_fn = preprocess.set_seed(42)
                      )

class SeqNegDataset(Dataset):
    """
    每条样本: (hist_seq, pos_item )
    hist_seq 已左侧 padding 到长度 max_len
    """
    def _sample_neg(self, u):
        """采 K 个对 u 不在正集合里的负样本"""
        negs = []
        pos_set = self.user_pos[u]
        while len(negs) < self.k_neg:
            # 先一次抽一大批候选，再过滤，加速
            need = self.k_neg - len(negs)
            pool = np.random.choice(
                self.item_pool,  # a: 只有出现过的 id
                size=need * 3,
                p=self.neg_dist  # p: 同长度概率向量
            )
            for n in pool:
                if n not in pos_set:
                    negs.append(n)
                    if len(negs) == self.k_neg:
                        break
        return negs
    def __init__(self, df, max_len,num_items, pad_idx, user_col, item_col, k_neg=1, neg_sampling='pop', alpha=0.75):
        super().__init__()
        self.users = df[user_col].to_numpy(dtype=np.int32)
        self.items = df[item_col].to_numpy(dtype=np.int32)
        self.num_items = num_items
        self.k_neg = k_neg

        # 1) 预构 user→正集合
        self.user_pos = (
            df.groupby(user_col)[item_col]
            .apply(lambda x: set(x.tolist()))
            .to_dict()
        )

        # 2) 负采样分布
        if neg_sampling == 'pop':  # 按 item 热度的 α 次幂
            pop = df[item_col].value_counts().sort_index()
            self.item_pool = pop.index.to_numpy()
            prob = (pop ** alpha) / (pop ** alpha).sum()
            self.neg_dist = np.asarray(prob, dtype=np.float64)
        else:  # uniform
            self.neg_dist = None

        self.max_len = max_len
        self.pad_idx = pad_idx

        self.users, self.inputs, self.targets, self.negs  = [], [], [], []
        for user, user_hist in df.groupby(user_col):
            seq = user_hist[item_col].tolist()


            for i in range(1, len(seq)):
                hist = seq[max(0, i - max_len): i]
                hist = [pad_idx] * (max_len - len(hist)) + hist
                neg = self._sample_neg(user)
                self.users.append(user)
                self.inputs.append(hist)
                self.targets.append(seq[i])  # 正样本
                self.negs.append(neg)


        self.users = np.asarray(self.users, dtype=np.int64)
        self.inputs  = np.asarray(self.inputs,  dtype=np.int64)
        self.targets = np.asarray(self.targets, dtype=np.int64)
        self.negs = np.asarray(self.negs, dtype=np.int64)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        hist = self.inputs[idx]
        pos  = self.targets[idx]
        neg = self.negs[idx]
        return (
            torch.tensor(hist, dtype=torch.long),    # (T,)
            torch.tensor(pos,  dtype=torch.long),    # ()
            torch.tensor(neg, dtype=torch.long),
        )
def build_seq_neg_loader(df, num_items, max_len, pad_idx, user_id, item_id ,batch_size=1024, shuffle=True, num_workers=2,k_neg=1, neg_sampling='pop', alpha=0.75):
    dataset = SeqNegDataset(df, max_len,num_items, pad_idx, user_id, item_id, k_neg=k_neg, neg_sampling=neg_sampling, alpha=alpha)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True,
                      drop_last=True,
                      worker_init_fn = preprocess.set_seed(42))