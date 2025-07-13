import random
from collections import Counter
import torch
import numpy as np


def hit_rate_at_k(ranked_list, true_item):
    return int(true_item in ranked_list)

def ndcg_at_k(ranked_list, true_item):
    if true_item in ranked_list:
        index = ranked_list.index(true_item)
        return 1 / np.log2(index + 2)
    else:
        return 0.0

def evaluate_random(test_loader, item_pool, top_k=10):
    hits, ndcgs = [], []
    for _, item_batch in test_loader:
        item_batch = item_batch.numpy()
        for true_item in item_batch:
            # 从全体物品中随机抽 top_k 个（包含或不包含 true_item）
            rec_list = random.sample(item_pool, top_k)
            hits.append(hit_rate_at_k(rec_list, true_item))
            ndcgs.append(ndcg_at_k(rec_list, true_item))
    return np.mean(hits), np.mean(ndcgs)


def evaluate_popular(test_loader, train_df, top_k=10):
    item_counts = Counter(train_df['item_id'].values)
    popular_items = [item for item, _ in item_counts.most_common(top_k)]

    hits, ndcgs = [], []
    for _, item_batch in test_loader:
        item_batch = item_batch.numpy()
        for true_item in item_batch:
            hits.append(hit_rate_at_k(popular_items, true_item))
            ndcgs.append(ndcg_at_k(popular_items, true_item))
    return np.mean(hits), np.mean(ndcgs)

def evaluate_model(test_loader, model, faiss_index, device, top_k=10):
    hits, ndcgs = [], []
    model.eval()

    for user_batch, item_batch in test_loader:
        user_batch = user_batch.to(device)
        item_batch = item_batch.cpu().numpy()  # true items

        with torch.no_grad():
            user_vecs = model.get_users_embedding(user_batch)
            user_vecs = user_vecs.cpu().numpy().astype(np.float32)

        # FAISS 批量 topK
        _, I = faiss_index.search(user_vecs, top_k)  # (B, K)
        topk_lists = I.tolist()

        for rec_list, true_item in zip(topk_lists, item_batch):
            hits.append(hit_rate_at_k(rec_list, true_item))
            ndcgs.append(ndcg_at_k(rec_list, true_item))

    return np.mean(hits), np.mean(ndcgs)

def evaluate_seq_model(test_loader, model, faiss_index, device, hist_tensors, top_k=10):
    hits, ndcgs = [], []
    model.eval()

    for user_batch, item_batch in test_loader:
        user_batch = user_batch.to(device)
        item_batch = item_batch.cpu().numpy()  # true items

        with torch.no_grad():
            seq = hist_tensors[user_batch]
            predict = model(seq)
            predict = predict.cpu().numpy().astype(np.float32)

        # FAISS 批量 topK
        _, I = faiss_index.search(predict, top_k)  # (B, K)
        topk_lists = I.tolist()

        for rec_list, true_item in zip(topk_lists, item_batch):
            hits.append(hit_rate_at_k(rec_list, true_item))
            ndcgs.append(ndcg_at_k(rec_list, true_item))

    return np.mean(hits), np.mean(ndcgs)