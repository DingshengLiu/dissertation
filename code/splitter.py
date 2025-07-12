def split(df, user_col='user', item_col='item', time_col='timestamp'):

    df = df.sort_values(by=[user_col, time_col])  # 按用户时间排序

    # 获取每个用户的最后一条记录作为 test
    test_df = df.groupby(user_col).tail(1)
    train_df = df.drop(index=test_df.index)

    # 过滤 test 中那些 user/item 不在 train 中的
    train_users = set(train_df[user_col])
    train_items = set(train_df[item_col])

    test_df = test_df[
        test_df[user_col].isin(train_users) &
        test_df[item_col].isin(train_items)
    ]

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)