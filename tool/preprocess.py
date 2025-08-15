import numpy as np
import pandas as pd
import lmdb
import torch
import random
def set_seed(seed=42):
    random.seed(seed)  # Python built-in random number generator
    np.random.seed(seed)  # Numpy random numbers
    torch.manual_seed(seed)  # Torch random numbers on CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs (multi-GPU)

    # Ensure the cuDNN algorithms return deterministic results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable automatic algorithm selection

    # Also set the seed for DataLoader workers (if num_workers > 0)
    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return seed_worker

def openAndSort(path,user_id,item_id,timestamp=None):
    dataset_pd = pd.read_csv(path)
    if timestamp is None:
     # 对于某些数据集，默认已经按照时序排序，不再对timestamp进行排序
     dataset_pd = dataset_pd.sort_values(by=[user_id])
    else:
     dataset_pd = dataset_pd.sort_values(by=[user_id,timestamp])

    # print
    num_users = dataset_pd[user_id].nunique()
    num_items = dataset_pd[item_id].nunique()
    num_records = len(dataset_pd)


    print("dataset base information：")
    print(f"- number of users：{num_users}")
    print(f"- number of items：{num_items}")
    print(f"- number of rows：{num_records}")

    return dataset_pd,num_users,num_items

def split(df, user_id, item_id, timestamp):
    # Get the last record of each user as the test set
    test_df = df.groupby(user_id).tail(1)
    train_df = df.drop(index=test_df.index)

    # Filter out test rows where the user/item is not in the train set
    train_users = set(train_df[user_id])
    train_items = set(train_df[item_id])

    # Ensure that all users/items in the test set also appear in the train set
    # This avoids having items that only appear in the test set without being seen in training
    test_df = test_df[
        test_df[user_id].isin(train_users) &
        test_df[item_id].isin(train_items)
    ]
    # Reset the index of the DataFrames so the index is consecutive
    # drop=True means the old index is discarded
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def split_with_val(df, user_id, item_id, timestamp):
    # 每个用户最后一条作为测试集
    test_df = df.groupby(user_id).tail(1)
    tmp_df = df.drop(index=test_df.index)

    # 每个用户倒数第二条作为验证集
    val_df = tmp_df.groupby(user_id).tail(1)
    train_df = tmp_df.drop(index=val_df.index)

    # 过滤验证集和测试集，确保其中的 user/item 出现在训练集
    train_users = set(train_df[user_id])
    train_items = set(train_df[item_id])

    val_df = val_df[val_df[user_id].isin(train_users) &
                    val_df[item_id].isin(train_items)]
    test_df = test_df[test_df[user_id].isin(train_users) &
                      test_df[item_id].isin(train_items)]

    # 包含验证集的训练集（最终训练用）
    train_all_df = pd.concat([train_df, val_df], ignore_index=True)

    # 保持接口习惯：返回 (train_df, val_df, test_df, train_all_df)
    return (
        train_df.reset_index(drop=True),        # 用于早停
        val_df.reset_index(drop=True),          # 验证集
        test_df.reset_index(drop=True),         # 测试集
        train_all_df.reset_index(drop=True)     # 最终训练集
    )
def load_lmdb_to_dict(lmdb_path, vector_dim=None, dtype=np.float32):
    env = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False, readahead=False)

    # 如果没有指定维度，尝试从 LMDB 中读取
    if vector_dim is None:
        with env.begin() as txn:
            dim_bytes = txn.get(b"__dim__")
            if dim_bytes:
                vector_dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
                print(f"Found stored dimension: {vector_dim}")
            else:
                print("No dimension info found in LMDB, will auto-detect from first item")

    raw_data = {}

    with env.begin() as txn:
        cursor = txn.cursor()
        for key_bytes, val_bytes in cursor:
            try:
                key_str = key_bytes.decode()
                if not key_str.isdigit():
                    continue
                key_int = int(key_str)
            except:
                continue

            raw_data[key_int] = bytes(val_bytes)  # 拷贝 buffer

    env.close()

    vectors = {}
    for k, val in raw_data.items():
        vec = np.frombuffer(val, dtype=dtype)

        # 自动检测维度
        if vector_dim is None:
            vector_dim = vec.size
            print(f"Auto-detected vector dimension: {vector_dim}")

        if vec.size != vector_dim:
            raise ValueError(f"Item {k} vector dim {vec.size} != {vector_dim}")
        vectors[k] = vec.copy()  # 拷贝防止潜在引用问题

    return vectors

def load_tensor_from_lmdb(lmdb_path, num_items, item_id_to_item, vector_dim=None, dtype=np.float32):
    cover_vec = load_lmdb_to_dict(lmdb_path, vector_dim ,dtype)
    # 冻结部分：构建 (num_items, 128) 的固定向量
    cover_emb_list = []
    for item_id in range(num_items):
        item = item_id_to_item[item_id]
        vec = cover_vec[item]  # numpy vector of shape (128,)
        cover_emb_list.append(vec)

    # 转成 tensor
    cover_emb_tensor = torch.tensor(np.stack(cover_emb_list), dtype=torch.float32)  # shape: (num_items, 128)
    return cover_emb_tensor