import csv
import re
import pickle

import numpy as np
from math import ceil

excluded_regex = re.compile(r"(?: ?)<EXCLUDE\/>(?: ?)(.*)")
np.random.seed(42)
rng = np.random.default_rng()


def load_all_csv_rows(file_path, encoding="utf8"):
    with open(file_path, encoding=encoding) as csvfile:
        reader = csv.reader(csvfile)
        all_rows = []
        for row in reader:
            all_rows.append(row)

    return all_rows


def load_all_domains():
    return load_all_csv_rows(
        "expert_annotations/annotated_domains.csv", encoding="UTF-8"
    )


def load_all_items():
    return load_all_csv_rows("expert_annotations/annotated_items.csv", encoding="UTF-8")


def gather_items(items):
    items = list(zip(*items))

    # FILTER EXCLUDED AND KEPT
    kept_items = []
    excluded_items = []
    for cluster_items in items:
        kept_items_cluster = []
        excluded_items_cluster = []

        for item in cluster_items:
            if len(item) > 0:
                match = excluded_regex.search(item)
                if match:
                    excluded_items_cluster.append(match.group(1))
                else:
                    kept_items_cluster.append(item)

        kept_items.append(kept_items_cluster)
        excluded_items.append(excluded_items_cluster)
    filtered_items = {"kept": kept_items, "excluded": excluded_items}

    # WRITE THEM BOTH TO SEPARATE FILES
    with open("expert_annotations/items/all.pck", "wb") as f:
        pickle.dump(filtered_items, f)

    # SEPARATE BOTH FILES IN TRAIN-VALID WITH 0.85:0.3 RATIO
    train_items = {"kept": [], "excluded": []}
    valid_items = {"kept": [], "excluded": []}

    for kept_items_cluster, excluded_items_cluster in zip(kept_items, excluded_items):
        n_valid_kept = ceil(0.3 * len(kept_items_cluster))
        valid_kept = rng.choice(len(kept_items_cluster), n_valid_kept)

        valid_kept_items_cluster = []
        train_kept_items_cluster = []
        for idx, item in enumerate(kept_items_cluster):
            if idx in valid_kept:
                valid_kept_items_cluster.append(item)
            else:
                train_kept_items_cluster.append(item)

        train_items["kept"].append(train_kept_items_cluster)
        valid_items["kept"].append(valid_kept_items_cluster)

        n_valid_excluded = ceil(0.3 * len(excluded_items_cluster))
        valid_excluded = rng.choice(len(excluded_items_cluster), n_valid_excluded)

        valid_excluded_items_cluster = []
        train_excluded_items_cluster = []
        for idx, item in enumerate(excluded_items_cluster):
            if idx in valid_excluded:
                valid_excluded_items_cluster.append(item)
            else:
                train_excluded_items_cluster.append(item)

        train_items["excluded"].append(train_excluded_items_cluster)
        valid_items["excluded"].append(valid_excluded_items_cluster)

    with open("expert_annotations/items/train.pck", "wb") as f:
        pickle.dump(train_items, f)
    with open("expert_annotations/items/valid.pck", "wb") as f:
        pickle.dump(valid_items, f)


def gather_domains(domains):
    domains = list(zip(*domains))

    # FILTER EXCLUDED AND KEPT
    kept_domains = []
    excluded_domains = []
    for cluster_domains in domains:
        kept_domains_cluster = []
        excluded_domains_cluster = []

        for item in cluster_domains:
            if len(item) > 0:
                match = excluded_regex.search(item)
                if match:
                    excluded_domains_cluster.append(match.group(1))
                else:
                    kept_domains_cluster.append(item)

        kept_domains.append(kept_domains_cluster)
        excluded_domains.append(excluded_domains_cluster)
    filtered_domains = {"kept": kept_domains, "excluded": excluded_domains}

    # WRITE THEM BOTH TO SEPARATE FILES
    with open("expert_annotations/domains/all.pck", "wb") as f:
        pickle.dump(filtered_domains, f)

    # SEPARATE BOTH FILES IN TRAIN-VALID WITH 0.7:0.3 RATIO
    train_domains = {"kept": [], "excluded": []}
    valid_domains = {"kept": [], "excluded": []}

    for kept_domains_cluster, excluded_domains_cluster in zip(
        kept_domains, excluded_domains
    ):
        n_valid_kept = ceil(0.3 * len(kept_domains_cluster))
        valid_kept = rng.choice(len(kept_domains_cluster), n_valid_kept)

        valid_kept_domains_cluster = []
        train_kept_domains_cluster = []
        for idx, item in enumerate(kept_domains_cluster):
            if idx in valid_kept:
                valid_kept_domains_cluster.append(item)
            else:
                train_kept_domains_cluster.append(item)

        train_domains["kept"].append(train_kept_domains_cluster)
        valid_domains["kept"].append(valid_kept_domains_cluster)

        n_valid_excluded = ceil(0.3 * len(excluded_domains_cluster))
        valid_excluded = rng.choice(len(excluded_domains_cluster), n_valid_excluded)

        valid_excluded_domains_cluster = []
        train_excluded_domains_cluster = []
        for idx, item in enumerate(excluded_domains_cluster):
            if idx in valid_excluded:
                valid_excluded_domains_cluster.append(item)
            else:
                train_excluded_domains_cluster.append(item)

        train_domains["excluded"].append(train_excluded_domains_cluster)
        valid_domains["excluded"].append(valid_excluded_domains_cluster)

    with open("expert_annotations/domains/train.pck", "wb") as f:
        pickle.dump(train_domains, f)
    with open("expert_annotations/domains/valid.pck", "wb") as f:
        pickle.dump(valid_domains, f)


def gather_all_annotations():
    gather_items(load_all_items())
    gather_domains(load_all_domains())


if __name__ == "__main__":
    gather_all_annotations()
