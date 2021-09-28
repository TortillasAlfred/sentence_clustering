import csv
import os
import pickle
from sklearn.metrics import f1_score
from itertools import product


def load_all_csv_rows(file_path, encoding="utf8"):
    with open(file_path, encoding=encoding) as csvfile:
        reader = csv.reader(csvfile)
        all_rows = []
        for row in reader:
            datum = row[0]
            if datum[-1] == " ":
                datum = datum[:-1]
            if datum[0] == " ":
                datum = datum[1:]

            all_rows.append(datum)

    return all_rows


def load_all_items():
    items = load_all_csv_rows("Full data items.csv", encoding="Windows-1252")
    items = [item for item in items if len(item.split()) > 2]

    return items


def get_annotated_data(set):
    with open(f"expert_annotations/{set}/all.pck", "rb") as f:
        annotated_data = pickle.load(f)

    return annotated_data


def get_supervised_score(labels, sentence_annotations, annotated_data):
    # Map sentences to cluster_idx for each kept/excluded cluster
    predicted_idxs = {}
    for key, clusters in annotated_data.items():
        predicted_idxs[key] = []

        for cluster in clusters:
            cluster_idxs = []

            for item in cluster:
                if item in labels:
                    cluster_idxs.append(labels[item])
                else:
                    # print("Hihi")
                    pass

            predicted_idxs[key].append(cluster_idxs)

    y_true, y_pred = [], []
    # Type 1 : 2 kept from same cluster, expect equals
    for cluster in predicted_idxs["kept"]:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                y_pred.append(cluster[i] == cluster[j])
                y_true.append(True)

    # Type 2 : 1 excluded & 1 kept from same cluster, expect not equals
    for kept_cluster, excluded_cluster in zip(
        predicted_idxs["kept"], predicted_idxs["excluded"]
    ):
        for kept in kept_cluster:
            for excluded in excluded_cluster:
                y_pred.append(kept == excluded)
                y_true.append(False)

    # Type 3 : 2 kept from different clusters, expect not equals
    for cluster_i in range(len(predicted_idxs["kept"])):
        for cluster_j in range(cluster_i + 1, len(predicted_idxs["kept"])):
            for item_i in predicted_idxs["kept"][cluster_i]:
                for item_j in predicted_idxs["kept"][cluster_j]:
                    y_pred.append(item_i == item_j)
                    y_true.append(False)

    f1_prev = f1_score(y_true, y_pred)

    y_true, y_pred = [], []

    for (idx1, cluster1), (idx2, cluster2) in product(
        sentence_annotations, sentence_annotations
    ):
        if cluster1 == cluster2:
            if idx1 in labels and idx2 in labels:
                y_true.append(True)
                y_pred.append(labels[idx1] == labels[idx2])
            else:
                y_true.append(True)
                y_pred.append(False)

    f1_dec3 = f1_score(y_true, y_pred)

    return f1_prev, f1_dec3


def get_item_index(item, sents):
    cleaned_item = " ".join(item.split())
    cleaned_item = cleaned_item.replace('"', "")
    cleaned_item = cleaned_item.replace(",", " ")
    cleaned_item = cleaned_item.replace("/", " / ")
    cleaned_item = cleaned_item.replace(".-", " .- ")
    cleaned_item = cleaned_item.replace(".", " . ")
    cleaned_item = cleaned_item.replace("   ", " ")
    cleaned_item = cleaned_item.replace("'", " ' ")
    cleaned_item = cleaned_item.replace("  ", " ")
    cleaned_item = cleaned_item.lower()

    if len(cleaned_item.split()) <= 2:
        return None
    while cleaned_item[-1] == " ":
        cleaned_item = cleaned_item[:-1]
    if cleaned_item in sents:
        return sents.index(cleaned_item)
    else:
        # print(cleaned_item)
        return None


def items_clustering():
    items = load_all_items()
    annotated_items = get_annotated_data("items")

    sents = [sent.replace('"', "") for sent in items]
    sents = [sent.replace(",", " ") for sent in sents]
    sents = [sent.replace("/", " / ") for sent in sents]
    sents = [sent.replace(".-", " .- ") for sent in sents]
    sents = [sent.replace(".", " . ") for sent in sents]
    sents = [sent.replace("'", " ' ") for sent in sents]
    sents = [sent.replace("\n", "") for sent in sents]
    sents = [sent.lower() for sent in sents]
    sents = [" ".join(sent.split()) for sent in sents]

    annotated_items_idxs = {}
    for key, key_items in annotated_items.items():
        annotated_items_idxs[key] = []

        for cluster in key_items:
            annotated_items_idxs[key].append([])
            for item in cluster:
                idx = get_item_index(item, sents)
                if idx:
                    annotated_items_idxs[key][-1].append(idx)

    with open(f"expert_annotations/dec3_expert_knowledge.pck", "rb") as f:
        dec3_data = pickle.load(f)

    sentence_annotations = []
    for index, samples in dec3_data.items():
        for item in samples:
            idx = get_item_index(item, sents)
            if idx:
                sentence_annotations.append((idx, index))

    results_folder = "./feb19_results"

    for clustering in os.listdir(results_folder):
        with open(os.path.join(results_folder, clustering), encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            clusters = {}
            for row in reader:
                for idx, item in enumerate(row):
                    if idx % 2 == 1:
                        continue
                    else:
                        idx = idx // 2
                    index = get_item_index(item, sents)
                    if index:
                        clusters[index] = idx

        print(
            f"{clustering} : {get_supervised_score(clusters, sentence_annotations, annotated_items_idxs)}"
        )
        print(len(clusters))


if __name__ == "__main__":
    items_clustering()
