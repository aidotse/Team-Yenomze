def split_train_val(data_list):
    split_point = int(len(data_list)*0.7)
    data_train = data_list[:split_point]
    data_val = data_list[split_point:]

    return data_train, data_val



