import os
import random
import time
import datetime

import torch
import numpy as np

from train.utils import grad_param, get_norm
from dataset.sampler import ParallelSampler_Test, ParallelSampler, task_sampler
from tqdm import tqdm
from termcolor import colored
from train.test import test
import torch.nn.functional as F


def train(train_data, val_data, model, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "tmp-runs", str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    opt = torch.optim.Adam(grad_param(model, ['E', 'D', 'clf']), lr=args.lr_g) # 包含生成器及分类器
    optD = torch.optim.Adam(grad_param(model, ['D_s']), lr=args.lr_d)

    print("\033[32m{}, Start training\033[0m".format(datetime.datetime.now()), flush=True)

    train_gen_val = ParallelSampler_Test(train_data, args, args.val_episodes)
    val_gen = ParallelSampler_Test(val_data, args, args.val_episodes)

    for ep in range(args.train_epochs):

        sampled_classes, source_classes = task_sampler(train_data, args)

        train_gen = ParallelSampler(train_data, args, sampled_classes, source_classes, args.train_episodes)

        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'E': [], 'D': [], 'D_s': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks,
                                 total=train_gen.num_episodes,
                                 ncols=80,
                                 leave=False,
                                 desc=colored('Training on train', 'yellow'))

        d_acc = 0
        for task in sampled_tasks:
            if task is None:
                break
            d_acc += train_one(task, model, opt, optD, args, grad)

        d_acc = d_acc / args.train_episodes
        print("---------------ep:" + str(ep) + " d_acc:" + str(d_acc) + "-----------")

        if ep % 10 == 0:

            acc, std = test(train_data, model, args, args.val_episodes, False, train_gen_val.get_epoch())
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now(),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)

        # Evaluate validation accuracy
        cur_acc, cur_std = test(val_data, model, args, args.val_episodes, False, val_gen.get_epoch())
        print(("{}, {:s} {:2d}, {:s} \033[36m{:s}{:>7.4f} ± {:>6.4f}\033[0m, \033[35m{:s}\033[0m {:s}{:>7.4f}, {:s}{:>7.4f}").format(
            datetime.datetime.now(),
            "ep", ep,
            "val  ",
            "acc:", cur_acc, cur_std,
            "train stats",
            "E_grad:", np.mean(np.array(grad['E'])),
            "clf_grad:", np.mean(np.array(grad['clf'])),
            ), flush=True
        )

        # Update the current best model if val acc is better
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_path = os.path.join(out_dir, str(ep))

            # save current model
            print("\033[34m{}, Save cur best model to {}\033[0m".format(
                datetime.datetime.now(),
                best_path))

            torch.save(model['E'].state_dict(), best_path + '.E')
            torch.save(model['D'].state_dict(), best_path + '.D')
            torch.save(model['D_s'].state_dict(), best_path + '.Dis')
            torch.save(model['clf'].state_dict(), best_path + '.clf')

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

    print("\033[33m{}, End of training. Restore the best weights\033[0m".format(
            datetime.datetime.now()),
            flush=True)

    # restore the best saved model
    model['E'].load_state_dict(torch.load(best_path + '.E'))
    model['D'].load_state_dict(torch.load(best_path + '.D'))
    model['D_s'].load_state_dict(torch.load(best_path + '.Dis'))
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    return

def InfoNCE(embed, tao=0.05, device='cuda'):
    label = torch.zeros(embed.size(0), device=device).long()
    similar = F.cosine_similarity(embed.unsqueeze(1), embed.unsqueeze(0), dim=2)
    similar = similar - torch.eye(embed.size(0), device=device) * 1e12
    similar = similar / tao
    loss = F.cross_entropy(similar, label)
    return loss

def train_one(task, model, opt, optD, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['E'].train()
    model['D'].train()
    model['D_s'].train()
    model['clf'].train()

    support, query, source = task

    # ***************update D**************
    opt.zero_grad()
    optD.zero_grad()

    # Embedding the document
    XS, XS_weight, XS_org, XS_e = model['E'](support, flag='support')
    XS_d = model['D'](XS_e)
    YS = support['label']

    XQ, XQ_weight, XQ_org, XQ_e = model['E'](query, flag='query')
    XQ_d = model['D'](XQ_e)
    YQ = query['label']
    YQ_dis = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)

    XSource, XSource_weight, XSource_org, XSource_e = model['E'](source, flag='query')
    XSource_d = model['D'](XSource_e)
    YSource_dis = torch.zeros(source['label'].shape, dtype=torch.long).to(source['label'].device)

    # decoder loss
    r_loss = F.mse_loss(XS_org, XS_d) + F.mse_loss(XQ_org, XQ_d) + F.mse_loss(XSource_org, XSource_d)

    XQ_logits = model['D_s'](XQ_weight)
    XSource_logits = model['D_s'](XSource_weight)

    dis_loss = F.cross_entropy(XQ_logits, YQ_dis) + F.cross_entropy(XSource_logits, YSource_dis)
    dis_loss.backward(retain_graph=True)
    grad['D_s'].append(get_norm(model['D_s']))
    optD.step()

    XQ_logits = model['D_s'](XQ_weight)
    XSource_logits = model['D_s'](XSource_weight)
    d_loss = F.cross_entropy(XQ_logits, YQ_dis) + F.cross_entropy(XSource_logits, YSource_dis)

    acc, d_acc, loss = model['clf'](XS, YS, XQ, YQ, XQ_logits, XSource_logits, YQ_dis, YSource_dis)

    a_loss = loss + r_loss - d_loss

    a_loss.backward(retain_graph=True)
    grad['E'].append(get_norm(model['E']))
    grad['D'].append(get_norm(model['D']))
    grad['clf'].append(get_norm(model['clf']))
    opt.step()

    return d_acc