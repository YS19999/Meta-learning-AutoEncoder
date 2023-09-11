import json

from classifier.get_classifier import get_classifier
from train.train import train
from train.test import test

from tools.tool import parse_args, set_seed

import dataset.loader as loader
from embedding.embedding import get_embedding

def main():

    args = parse_args()

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    # initialize model
    model = {}
    model["E"], model["D"], model['D_s'] = get_embedding(vocab, args) # 得到生成器和判别器
    model["clf"] = get_classifier(model["D"].ebd_embedding, args) # 得到分类器

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train(train_data, val_data, model, args)

    test_acc, test_std = test(test_data, model, args, args.test_episodes)

    if args.result_path:

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
        }

        save_args = ['test_acc', 'test_std', 'classifier', 'dataset', 'embedding', "n_test_class", "n_train_class", "n_val_class", 'way', 'query', 'shot']

        for attr, value in sorted(args.__dict__.items()):
            if attr in save_args:
                result[attr] = value

        with open(args.result_path, "a", encoding='UTF-8') as f:
            result = json.dumps(result, ensure_ascii=False)
            f.writelines(result + '\n')


if __name__ == '__main__':
    main()