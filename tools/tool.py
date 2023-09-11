import argparse

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="MetaAE FOR FEW-SHOT LEARNING")

    parser.add_argument("--data_path", type=str, default="data/huffpost.json")
    parser.add_argument("--dataset", type=str, default="huffpost")

    parser.add_argument("--n_train_class", type=int, default=20,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=16,
                        help="number of meta-test classes")

    parser.add_argument("--n_workers", type=int, default=10,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")

    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=5,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")

    parser.add_argument("--train_epochs", type=int, default=50,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    parser.add_argument("--wv_path", type=str,
                        default='word2vec',
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default='wiki.en.vec',
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=False,
                        help=("Finetune embedding during meta-training"))


    parser.add_argument("--bert", type=str, default=False,
                        help=("using bert embedding."))
    parser.add_argument("--embedding", type=str, default="metaae",
                        help=("document embedding method."))
    parser.add_argument("--classifier", type=str, default="r2d2",
                        help=("classifier. [mlp, proto, rn, routing, lrd2, r2d2]"))

    # lrd2 configuration
    parser.add_argument("--lrd2_num_iters", type=int, default=5,
                        help=("num of Newton steps for LRD2"))

    # proto configuration
    parser.add_argument("--proto_hidden", nargs="+", type=int,
                        default=[128, 128],
                        help=("hidden dimension of the proto-net"))

    parser.add_argument("--mlp_hidden", nargs="+", type=int, default=[300, 10],
                        help=("hidden dimension of the proto-net"))

    parser.add_argument("--induct_hidden_dim", type=int, default=100,
                        help=("tensor layer dim of induction network's relation"))
    parser.add_argument("--induct_iter", type=int, default=3,
                        help=("num of routings"))
    parser.add_argument("--induct_att_dim", type=int, default=64,
                        help=("attention projection dim of induction network"))


    parser.add_argument("--seed", type=int, default=330, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--patience", type=int, default=10, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=0,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="train",
                        help=("Running mode."
                              "Options: [train, test]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="results.json")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")

    parser.add_argument("--lr_g", type=float, default=1e-3, help="learning rate of meta-encoder, meta-decoder")
    parser.add_argument("--lr_d", type=float, default=1e-3, help="learning rate of meta-discriminator")
    parser.add_argument("--train_mode", type=str, default=None, help="you can choose t_add_v or None")
    parser.add_argument("--path_drawn_data", type=str, default="reuters_False_data.json", help="path_drawn_data")

    return parser.parse_args()


def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def load_model_state_dict(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    keys = []
    for k, v in pretrained_dict.items():
           keys.append(k)

    i = 0

    print("_____________pretrain_parameters______________________________")
    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            print(model_dict[k])
            i = i + 1
        # print(model_dict[k])
    print("___________________________________________________________")
    model.load_state_dict(model_dict)
    return model