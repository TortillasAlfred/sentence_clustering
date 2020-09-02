import pickle

from sentence_transformers import (
    SentenceTransformer,
    SentencesDataset,
    InputExample,
    evaluation,
    losses,
)
from torch.utils.data import DataLoader


def get_examples_from_data(data):
    examples = []

    # Type 1 : 2 kept from same cluster, label = 1
    for cluster in data["kept"]:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                examples.append([cluster[i], cluster[j], 1])

    # Type 2 : 1 excluded & 1 kept from same cluster, label = 0
    for kept_cluster, excluded_cluster in zip(data["kept"], data["excluded"]):
        for kept in kept_cluster:
            for excluded in excluded_cluster:
                examples.append([kept, excluded, 0])

    # Type 3 : 2 kept from different clusters, label = 0
    for cluster_i in range(len(data["kept"])):
        for cluster_j in range(cluster_i + 1, len(data["kept"])):
            for item_i in data["kept"][cluster_i]:
                for item_j in data["kept"][cluster_j]:
                    examples.append([item_i, item_j, 0])

    return examples


def extract_examples(set):
    with open(f"expert_annotations/{set}/train.pck", "rb") as f:
        train_data = pickle.load(f)
    with open(f"expert_annotations/{set}/valid.pck", "rb") as f:
        valid_data = pickle.load(f)

    train_examples = get_examples_from_data(train_data)
    valid_examples = get_examples_from_data(valid_data)

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
    evaluator = evaluation.EmbeddingSimilarityEvaluator(*zip(*valid_examples))

    return train_examples, evaluator


def main(model_name, loss, batch_size):
    train_examples, evaluator = get_experimental_setup()
    model = SentenceTransformer(model_name)
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    model.fit(
        [(train_dataloader, loss(model=model))],
        evaluator=evaluator,
        evaluation_steps=400,
        warmup_steps=4000,
        epochs=2,
        output_path=f"./best_finetuned_models/{model_name}/{str(loss)}/",
    )


if __name__ == "__main__":
    model = "distilbert-base-uncased"
    loss = losses.ContrastiveLoss
    batch_size = 64

    main(model, loss, batch_size)
