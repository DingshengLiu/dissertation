import numpy as np
import pandas as pd
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

    # 获取每个用户的最后一条记录作为 test
    test_df = df.groupby(user_id).tail(1)
    train_df = df.drop(index=test_df.index)

    # 过滤 test 中那些 user/item 不在 train 中的
    train_users = set(train_df[user_id])
    train_items = set(train_df[item_id])

    # 确保测试集中出现的用户/物品都在训练集中出现过，避免某个物品仅出现在测试集中，没有在训练集中得到过训练
    test_df = test_df[
        test_df[user_id].isin(train_users) &
        test_df[item_id].isin(train_items)
    ]
    # .reset_index重置 df 的索引，使得不连续的索引重新排列整齐，drop=True表明旧的索引不再保留
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)