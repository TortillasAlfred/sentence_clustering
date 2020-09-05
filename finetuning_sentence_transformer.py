import pickle

from sentence_transformers import (
    SentenceTransformer,
    SentencesDataset,
    ParallelSentencesDataset,
    InputExample,
    evaluation,
    losses,
)
import csv
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from itertools import product
from argparse import ArgumentParser

np.random.seed(37)
rng = np.random.default_rng()


def get_examples_from_data(data, train):
    examples = []

    # Type 1 : 2 kept from same cluster, label = 1
    type1 = []
    for cluster in data["kept"]:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                type1.append([cluster[i], cluster[j], 1])

    # Type 2 : 1 excluded & 1 kept from same cluster, label = 0
    type2 = []
    for kept_cluster, excluded_cluster in zip(data["kept"], data["excluded"]):
        for kept in kept_cluster:
            for excluded in excluded_cluster:
                type2.append([kept, excluded, 0])

    # Type 3 : 2 kept from different clusters, label = 0
    type3 = []
    for cluster_i in range(len(data["kept"])):
        for cluster_j in range(cluster_i + 1, len(data["kept"])):
            for item_i in data["kept"][cluster_i]:
                for item_j in data["kept"][cluster_j]:
                    type3.append([item_i, item_j, 0])

    if train:
        n_repeats_positive = int((len(type2) + len(type3)) / len(type1))
        examples.extend(type1 * n_repeats_positive)
    else:
        examples.extend(type1)

    examples.extend(type2)
    examples.extend(type3)

    return examples


def extract_examples(set):
    with open(f"expert_annotations/{set}/train.pck", "rb") as f:
        train_data = pickle.load(f)
    with open(f"expert_annotations/{set}/valid.pck", "rb") as f:
        valid_data = pickle.load(f)

    train_examples = get_examples_from_data(train_data, train=True)
    valid_examples = get_examples_from_data(valid_data, train=False)

    return train_examples, valid_examples


def get_binary_experimental_setup():
    # Items
    train_items, valid_items = extract_examples("items")

    # Domains
    train_domains, valid_domains = extract_examples("domains")

    # Regroup items and domains together
    train_examples = train_items + train_domains
    valid_examples = valid_items + valid_domains

    print(
        f"{len(train_examples)} training examples to {len(valid_examples)} valid examples"
    )

    # Postprocess train examples to correct format
    train_examples = [
        InputExample(texts=[sent1, sent2], label=label)
        for (sent1, sent2, label) in train_examples
    ]

    # Get evaluator from valid data
    evaluator = evaluation.BinaryClassificationEvaluator(
        *zip(*valid_examples), batch_size=128
    )

    return train_examples, evaluator


def get_mse_experimental_setup(student_model, teacher_model):
    dataset = ParallelSentencesDataset(student_model, teacher_model, batch_size=16)
    for _ in range(200):
        dataset.load_data("all_data.csv")

    return dataset


def get_cosine_experimental_setup(teacher_model):
    all_data = []
    with open("all_data.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            all_data.append(row[0])

    # Get embeddings
    target_embeddings = teacher_model.encode(all_data, convert_to_numpy=True)

    # Get cosine distances
    dists = cosine_distances(target_embeddings, target_embeddings)

    # Build dataset
    all_pairs = list(product(np.arange(len(all_data)), np.arange(len(all_data))))
    sampled_pairs = rng.choice(all_pairs, 500000, replace=False)

    cosine_examples = []
    for i, j in sampled_pairs:
        cosine_examples.append(
            InputExample(texts=[all_data[i], all_data[j]], label=dists[i, j])
        )

    return cosine_examples


def main(model_name, batch_size):
    model = SentenceTransformer(model_name)
    teacher_model = SentenceTransformer(model_name)

    # Binary loss setting
    binary_train_examples, evaluator = get_binary_experimental_setup()
    binary_dataset = SentencesDataset(binary_train_examples, model)
    binary_dataloader = DataLoader(
        binary_dataset, shuffle=True, batch_size=batch_size, num_workers=1
    )

    # MSE loss setting
    # mse_dataset = get_mse_experimental_setup(model, teacher_model)
    # mse_dataloader = DataLoader(
    #     mse_dataset, shuffle=True, batch_size=batch_size, num_workers=1
    # )

    # Cosine loss setting
    cosine_train_examples = get_cosine_experimental_setup(teacher_model)
    cosine_dataset = SentencesDataset(cosine_train_examples, model)
    cosine_dataloader = DataLoader(
        cosine_dataset, shuffle=True, batch_size=batch_size, num_workers=1
    )

    # Training
    model.fit(
        [
            (binary_dataloader, losses.OnlineContrastiveLoss(model=model)),
            # (mse_dataloader, losses.MSELoss(model=model)),
            (cosine_dataloader, losses.CosineSimilarityLoss(model=model)),
        ],
        evaluator=evaluator,
        evaluation_steps=2500,
        warmup_steps=5000,
        epochs=2,
        output_path=f"./best_finetuned_models/{model_name}/",
        output_path_ignore_not_empty=True,
    )


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--model", type=int)
    options = argument_parser.parse_args()

    torch.multiprocessing.set_start_method("forkserver", force=True)
    models = [
        "bert-base-nli-stsb-mean-tokens",
        "bert-base-nli-mean-tokens",
        "distilbert-base-nli-stsb-mean-tokens",
        "distilbert-base-nli-mean-tokens",
    ]
    batch_size = 16

    main(models[options.model], batch_size)
