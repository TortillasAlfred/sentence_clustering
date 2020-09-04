import csv
import os
import numpy as np

from sentence_transformers import SentenceTransformer, losses
from collections import defaultdict
from itertools import product
from sklearn.metrics.pairwise import cosine_distances


def icf_terms():
    with open("./icf_terms.txt", "r") as f:
        data = f.read().splitlines()

    icf_dict = defaultdict(set)

    for d in data:
        icf_dict[len(d.split())].add(d)

    return icf_dict


def load_all_csv_rows(file_path, encoding="utf8"):
    with open(file_path, encoding=encoding) as csvfile:
        reader = csv.reader(csvfile)
        all_rows = []
        for row in reader:
            all_rows.append(row[0])

    return all_rows


def load_all_domains():
    return load_all_csv_rows("Full data domains.csv", encoding="ISO-8859-1")


def load_all_items():
    return load_all_csv_rows("Full data items.csv", encoding="Windows-1252")


def main(model_name, loss, items):
    os.makedirs(
        f"embeddings_quality_output/{model_name}/{str(loss)}",
        exist_ok=True,
    )
    model = SentenceTransformer(f"./best_finetuned_models/{model_name}/{str(loss)}/")

    # (item, embedding) collection
    item_embeddings = model.encode(
        items, batch_size=128, convert_to_numpy=True, num_workers=10
    )
    items_dict = {
        item: item_embedding for item, item_embedding in zip(items, item_embeddings)
    }

    # Load pertinent items
    with open("pertinent_sents.txt", "r") as f:
        pertinent_sents = f.read().splitlines()

    # Compute cosinus distances to pertinent items
    pertinent_embeddings = np.stack(
        [items_dict[pertinent_sent] for pertinent_sent in pertinent_sents]
    )
    cos_dists = cosine_distances(pertinent_embeddings, item_embeddings)

    # Write to output csv like model/loss/item.csv, with [item, cosine] rows
    for pertinent_sent, cos_distances in zip(pertinent_sents, cos_dists):
        ordered_args = cos_distances.argsort()
        with open(
            f"embeddings_quality_output/{model_name}/{str(loss)}/{pertinent_sent}.csv",
            "w",
        ) as f:
            writer = csv.writer(f)

            for idx in ordered_args:
                writer.writerow([items[idx], cos_distances[idx]])


if __name__ == "__main__":
    models = [
        "bert-base-nli-stsb-mean-tokens",
        "bert-base-nli-mean-tokens",
        "distilbert-base-nli-stsb-mean-tokens",
        "distilbert-base-nli-mean-tokens",
    ]
    loss_functions = [losses.OnlineContrastiveLoss, losses.ContrastiveLoss]
    items = load_all_items()
    items = [" ".join(item.split()) for item in items]

    for model, loss in product(models, loss_functions):
        main(model, loss, items)
