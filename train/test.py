import datetime

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from dataset.sampler import ParallelSampler_Test


def test_one(task, model, args):

    support, query = task

    # Embedding the document
    XS, XS_weight, XS_org, XS_e = model['E'](support, flag='support')
    YS = support['label']

    XQ, XQ_weight, XQ_org, XQ_e = model['E'](query, flag='query')
    YQ = query['label']
    YQ_dis = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)

    XSource, XSource_weight, XSource_org, XSource_e = model['E'](query, flag='query')
    YSource_dis = torch.zeros(query['label'].shape, dtype=torch.long).to(query['label'].device)

    XQ_logits = model['D_s'](XQ_weight)
    XSource_logits = model['D_s'](XSource_weight)

    # Apply the classifier
    acc, _, _ = model['clf'](XS, YS, XQ, YQ, XQ_logits, XSource_logits, YQ_dis, YSource_dis)

    return acc


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):

    model['E'].eval()
    # model['D'].eval()
    model['D_s'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler_Test(test_data, args, num_episodes).get_epoch()

    acc = []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80, leave=False, desc=colored('Testing on val', 'yellow'))

    for task in sampled_tasks:
        acc1 = test_one(task, model, args)
        acc.append(acc1)

    acc = np.array(acc)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
            datetime.datetime.now(),
            colored("test acc mean", "blue"),
            np.mean(acc),
            colored("test std", "blue"),
            np.std(acc),
        ), flush=True)

    return np.mean(acc), np.std(acc)