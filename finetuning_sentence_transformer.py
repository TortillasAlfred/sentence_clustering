import pickle

from sentence_transformers import (
    SentenceTransformer,
    SentencesDataset,
    InputExample,
    evaluation,
    losses,
)
from itertools import product
from torch.utils.data import DataLoader


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


def get_experimental_setup():
    # Items
    train_items, valid_items = extract_examples("items")

    # Domains
    train_domains, valid_domains = extract_examples("domains")

    # Regroup items and domains together
    train_examples = train_items + train_domains
    valid_examples = valid_items + valid_domains

    # Postprocess train examples to correct format
    train_examples = [
        InputExample(texts=[sent1, sent2], label=label)
        for (sent1, sent2, label) in train_examples
    ]

    # Get evaluator from valid data
    evaluator = evaluation.BinaryClassificationEvaluator(
        *zip(*valid_examples), batch_size=64
    )

    return train_examples, evaluator


def main(model_name, loss, batch_size):
    train_examples, evaluator = get_experimental_setup()
    model = SentenceTransformer(model_name)
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=8
    )

    model.fit(
        [(train_dataloader, loss(model=model))],
        evaluator=evaluator,
        evaluation_steps=200,
        warmup_steps=2000,
        epochs=1,
        output_path=f"./best_finetuned_models/{model_name}/{str(loss)}/",
        output_path_ignore_not_empty=True,
    )


if __name__ == "__main__":
    models = [
        "bert-base-nli-stsb-mean-tokens",
        "bert-base-nli-mean-tokens",
        "distilbert-base-nli-stsb-mean-tokens",
        "distilbert-base-nli-mean-tokens",
    ]
    loss_functions = [losses.OnlineContrastiveLoss]
    batch_size = 64

    for model, loss in product(models, loss_functions):
        main(model, loss, batch_size)
