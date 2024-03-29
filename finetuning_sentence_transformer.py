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
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import product
from argparse import ArgumentParser

np.random.seed(37)
rng = np.random.default_rng()


def get_examples_from_data(data):
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

    n_repeats_positive = int((len(type2) + len(type3)) / len(type1))
    examples.extend(type1 * n_repeats_positive)

    examples.extend(type2)
    examples.extend(type3)

    return examples


def extract_examples(set):
    with open(f"expert_annotations/{set}/train.pck", "rb") as f:
        train_data = pickle.load(f)
    with open(f"expert_annotations/{set}/valid.pck", "rb") as f:
        valid_data = pickle.load(f)

    train_examples = get_examples_from_data(train_data)
    train_dec3_examples = get_dec3_examples()
    train_examples.extend(train_dec3_examples)
    valid_examples = get_examples_from_data(valid_data)

    return train_examples, valid_examples


def extract_classif_examples():
    with open(f"expert_annotations/dec3_expert_knowledge.pck", "rb") as f:
        data = pickle.load(f)

    examples = []

    # Rebalance
    items_per_label = [len(d) for d in data.values()]
    target_num = max(items_per_label)
    mults_per_label = [int(target_num / l) for l in items_per_label]

    for (label, samples), mult in zip(data.items(), mults_per_label):
        for sample in samples:
            for _ in range(mult):
                examples.append((sample, label))

    num_labels = len(data)

    return examples, num_labels


def get_dec3_examples():
    with open(f"expert_annotations/dec3_expert_knowledge.pck", "rb") as f:
        data = pickle.load(f)

    examples = []

    # Type 1 : 2 kept from same cluster, label = 1
    type1 = []
    for cluster in data.values():
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                type1.append([cluster[i], cluster[j], 1])

    # Type 3 : 2 kept from different clusters, label = 0
    type3 = []
    for cluster_i in range(len(data)):
        for cluster_j in range(cluster_i + 1, len(data)):
            for item_i in data[cluster_i]:
                for item_j in data[cluster_j]:
                    type3.append([item_i, item_j, 0])

    n_repeats_positive = int((len(type3)) / len(type1))
    examples.extend(type1 * n_repeats_positive)

    examples.extend(type3)

    return examples


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


def get_softmax_experimental_setup():
    train_examples, num_labels = extract_classif_examples()

    # Postprocess train examples to correct format
    train_examples = [
        InputExample(texts=[sent, sent], label=label)
        for (sent, label) in train_examples
    ] * 100

    return train_examples, num_labels


def get_mse_experimental_setup(student_model, teacher_model):
    dataset = ParallelSentencesDataset(student_model, teacher_model)
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
    dists = cosine_similarity(target_embeddings, target_embeddings)

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
        binary_dataset, shuffle=True, batch_size=batch_size, num_workers=0
    )

    # MSE loss setting
    mse_dataset = get_mse_experimental_setup(model, teacher_model)
    mse_dataloader = DataLoader(
        mse_dataset, shuffle=True, batch_size=batch_size, num_workers=0
    )

    # Cosine loss setting
    cosine_train_examples = get_cosine_experimental_setup(model)
    cosine_dataset = SentencesDataset(cosine_train_examples, model)
    cosine_dataloader = DataLoader(
        cosine_dataset, shuffle=True, batch_size=batch_size, num_workers=0
    )

    # Softmax prediction setting
    softmax_train_examples, num_labels = get_softmax_experimental_setup()
    softmax_dataset = SentencesDataset(softmax_train_examples, model)
    softmax_dataloader = DataLoader(
        softmax_dataset, shuffle=True, batch_size=batch_size, num_workers=0
    )

    # Training
    model.fit(
        [
            (binary_dataloader, ContrastiveLoss(model=model, weight=0.5)),
            (
                softmax_dataloader,
                SoftmaxLoss(
                    model=model,
                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                    num_labels=num_labels,
                    weight=1.0,
                ),
            ),
            (mse_dataloader, MSELoss(model=model, weight=0.25)),
            (cosine_dataloader, CosineSimilarityLoss(model=model, weight=0.25)),
        ],
        evaluator=evaluator,
        evaluation_steps=1000,
        warmup_steps=1000,
        epochs=1,
        optimizer_params={"lr": 1e-5, "eps": 1e-8, "correct_bias": False},
        output_path=f"./best_finetuned_models/{model_name}/",
        output_path_ignore_not_empty=True,
    )


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        margin=0.5,
        size_average=True,
        weight=1.0,
    ):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model
        self.size_average = size_average
        self.weight = weight

    def forward(self, sentence_features, labels):
        reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        assert len(reps) == 2
        rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
            labels.float() * distances.pow(2)
            + (1 - labels).float()
            * torch.nn.functional.relu(self.margin - distances).pow(2)
        )
        loss = losses.mean() if self.size_average else losses.sum()
        loss = loss * self.weight
        return loss


class SoftmaxLoss(nn.Module):
    def __init__(self, model, sentence_embedding_dimension, num_labels, weight=1.0):
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.classifier = nn.Linear(sentence_embedding_dimension, num_labels)
        self.weight = weight

    def forward(self, sentence_features, labels):
        reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        rep_a, rep_b = reps

        vectors_concat = []

        vectors_concat.append(rep_a)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(output, labels.view(-1))
        loss = self.weight * loss
        return loss


class OnlineContrastiveLoss(nn.Module):
    def __init__(
        self,
        model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        margin: float = 0.5,
        weight=1.0,
    ):
        super(OnlineContrastiveLoss, self).__init__()
        self.model = model
        self.margin = margin
        self.distance_metric = distance_metric
        self.weight = weight

    def forward(self, sentence_features, labels, size_average=False):
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = (
            torch.nn.functional.relu(self.margin - negative_pairs).pow(2).sum()
        )
        loss = positive_loss + negative_loss
        loss = self.weight * loss
        return loss


class MSELoss(nn.Module):
    def __init__(self, model, weight=1.0):
        super(MSELoss, self).__init__()
        self.model = model
        self.weight = weight

    def forward(self, sentence_features, labels):
        rep = self.model(sentence_features[0])["sentence_embedding"]
        loss_fct = nn.MSELoss()
        loss = loss_fct(rep, labels)
        loss = self.weight * loss
        return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(
        self,
        model,
        loss_fct=nn.MSELoss(),
        cos_score_transformation=nn.Identity(),
        weight=1.0,
    ):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.weight = weight

    def forward(self, sentence_features, labels):
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        output = self.cos_score_transformation(
            torch.cosine_similarity(embeddings[0], embeddings[1])
        )
        loss = self.loss_fct(output, labels.view(-1))
        loss = self.weight * loss
        return loss


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--model", type=int)
    options = argument_parser.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)

    models = [
        "bert-base-nli-stsb-mean-tokens",
        "bert-base-nli-mean-tokens",
        "distilbert-base-nli-stsb-mean-tokens",
        "distilbert-base-nli-mean-tokens",
    ]
    batch_size = 32

    main(models[options.model], batch_size)
