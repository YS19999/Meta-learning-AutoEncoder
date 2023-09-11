import torch
import datetime


def tprint(s):

    print('{}: {}'.format(datetime.datetime.now(), s), flush=True)


def to_tensor(data, cuda, exclude_keys=[]):

    for key in data.keys():
        if key in exclude_keys:
            continue

        data[key] = torch.from_numpy(data[key])
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data


def select_subset(old_data, new_data, keys, idx, max_len=None):

    for k in keys:
        new_data[k] = old_data[k][idx]
        if max_len is not None and len(new_data[k].shape) > 1:
            new_data[k] = new_data[k][:, :max_len]

    return new_data
