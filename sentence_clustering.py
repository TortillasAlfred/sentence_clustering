import csv

from torchtext.vocab import Vocab, Vectors
from collections import Counter, OrderedDict, defaultdict
from itertools import product
import os
import pickle
import shutil
from sentence2vec import sentence2vec
from copy import deepcopy
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
import fasttext
import re
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})


def icf_terms():
    with open("./icf_terms.txt", "r") as f:
        data = f.read().splitlines()

    icf_dict = defaultdict(set)

    for d in data:
        icf_dict[len(d.split())].add(d)

    return icf_dict


def load_all_csv_rows(file_path):
    with open(file_path, encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)
        all_rows = []
        for row in reader:
            all_rows.append(row[0])

    return all_rows


def load_all_domains():
    return load_all_csv_rows("Domain-mobility-test-RA1.csv")


def load_all_items():
    return load_all_csv_rows("Mobility_item_test_RA.csv")


def create_folder_for_config(config, pre_config, base_path):
    save_path = base_path

    for key, val in pre_config.items():
        save_path = os.path.join(save_path, f"{key}::{val}")

    for key, val in config.items():
        save_path = os.path.join(save_path, f"{key}::{val}")

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path)

    return save_path


def get_vocab_counter(sents, word_filtering):
    raw_sents = deepcopy(sents)
    sents = [sent.replace('"', "") for sent in sents]
    sents = [sent.replace(",", " ") for sent in sents]
    sents = [sent.replace("/", " / ") for sent in sents]
    sents = [sent.replace(".-", " .- ") for sent in sents]
    sents = [sent.replace(".", " . ") for sent in sents]
    sents = [sent.replace("'", " ' ") for sent in sents]
    sents = [sent.lower() for sent in sents]

    if word_filtering == "none":
        sents = [sent.split() for sent in sents]
        sents = [(raw_sents[i], sent) for i, sent in enumerate(sents) if len(sent) > 0]
    elif "automatic_filtering" in word_filtering:
        filter_freq = int(word_filtering.split("_")[-1])
        sents = [sent.split() for sent in sents]
        words = [word for sent in sents for word in sent]

        vocab = Counter(words)
        words_too_frequent = set(
            [word for word, freq in vocab.items() if freq > filter_freq]
        )

        must_keep_terms = icf_terms()

        regex = re.compile(r"( (?:no|yes|not|grade) .*)")

        def process_sent(sents):
            raw_sent, sent = sents
            if len(sent) > 2:
                sent = regex.sub("", raw_sent)

                sent = sent.split()

                sent_too_freqs = [word in words_too_frequent for word in sent]

                for i in range(len(sent)):
                    for r in range(1, len(sent) - i):
                        sub_term = " ".join(sent[i : i + r])
                        if sub_term in must_keep_terms[r]:
                            sent_too_freqs[i : i + r] = [False] * r

                idxs_to_remove = []
                running_idxs_group = []

                for i in range(len(sent)):
                    if sent_too_freqs[i]:
                        running_idxs_group.append(i)
                    else:
                        if len(running_idxs_group) >= 3:
                            idxs_to_remove.extend(running_idxs_group)
                        running_idxs_group = []

                if len(running_idxs_group) >= 3:
                    idxs_to_remove.extend(running_idxs_group)

                sent = [
                    word for idx, word in enumerate(sent) if idx not in idxs_to_remove
                ]

                if len(sent) > 0:
                    return (raw_sent, sent)
                else:
                    return None
            else:
                return (raw_sent, sent)

        sents = map(process_sent, zip(raw_sents, sents))
        sents = list(filter(None, sents))

    elif word_filtering == "word_groups":
        with open("group_words_to_remove.txt", "r") as f:
            words_to_remove = f.read().splitlines()
            words_to_remove = set(words_to_remove)
            words_to_remove = [group.split() for group in words_to_remove]

        regex = re.compile(r"( (?:no|yes|not|grade) .*)")

        def process_sent(sents):
            raw_sent, sent = sents
            sent = regex.sub("", sent)

            def present_group(group, sent):
                def present_at_idx(group, sent, sent_idx):
                    for group_idx in range(len(group)):
                        if (
                            sent_idx + group_idx >= len(sent)
                            or sent[sent_idx + group_idx] != group[group_idx]
                        ):
                            return False

                    return True

                for sent_idx in range(len(sent)):
                    if present_at_idx(group, sent, sent_idx):
                        return True

                return False

            sent = sent.split()
            present_groups = [
                group for group in words_to_remove if present_group(group, sent)
            ]
            words = list(set([w for group in present_groups for w in group]))

            sent = [w for w in sent if w not in words]

            if len(sent) > 0:
                return (raw_sent, sent)
            else:
                return None

        sents = map(process_sent, zip(raw_sents, sents))
        sents = list(filter(None, sents))
    else:
        raise ValueError(f"Incorrect word filtering :{word_filtering}")

    words = [word for sent in sents for word in sent[1]]

    return Counter(words), sents


def preprocess(sents, word_filtering, vectors):
    vocab_counter, sents = get_vocab_counter(sents, word_filtering)

    vocab = Vocab(vocab_counter, vectors=None, specials=[],)

    return vocab, sents


def get_clustering_obj(method, clusters):
    if "kmeans" in method:
        return KMeans(
            n_clusters=clusters,
            random_state=42,
            n_init=20,
            n_jobs=-1,
            max_iter=1000,
            tol=1e-5,
        )
    elif method == "spectral":
        return SpectralClustering(
            n_clusters=clusters, random_state=42, n_init=20, n_jobs=-1
        )
    elif method == "nearest_neighbor":
        return None
    else:
        raise ValueError("Unknown clustering method")


def run_clustering(
    method,
    clusters,
    sentences,
    sentence_vectors,
    base_path,
    config,
    pre_config,
    save_path,
):
    clustering_obj = get_clustering_obj(method, clusters)

    if "kmeans_icf" in method:
        icf_sents = [term for term_set in icf_terms().values() for term in term_set]
        vocab, icf_sents = preprocess(icf_sents, "none", pre_config["reduce_method"])
        icf_sent_embeddings = sentence_vectorize(
            pre_config["reduce_method"], icf_sents, vocab
        )
        icf_sent_embeddings = reduce_dim(icf_sent_embeddings, config["reduced_dim"])

        n_sents = len(sentences)

        icf_weight = float(method.split("_")[-1])
        total_sents = np.vstack((sentence_vectors, icf_sent_embeddings))
        sent_weights = [0] * len(total_sents)
        sent_weights[:n_sents] = [(1.0 - icf_weight) / len(sentences)] * n_sents
        sent_weights[n_sents:] = [icf_weight / len(icf_sents)] * (
            len(total_sents) - n_sents
        )

        labels = clustering_obj.fit_predict(total_sents, sample_weight=sent_weights)[
            :n_sents
        ]
    elif method == "nearest_neighbor":
        load_path = save_path.replace("items", "domains")
        load_path = load_path.replace("method::nearest_neighbor", "method::kmeans")
        with open(os.path.join(load_path, "clusters_centers.pck"), "rb") as f:
            clusters_centers = pickle.load(f)

        clusters_centers = np.asarray(clusters_centers)

        dists = cdist(sentence_vectors, clusters_centers, metric="cosine")
        labels = np.argmin(dists, axis=-1)
    else:
        labels = clustering_obj.fit_predict(sentence_vectors)

    if len(set(labels)) == 0:
        score = 0.0
    else:
        score = silhouette_score(sentence_vectors, labels, metric="cosine")

    return score, labels


def get_rows(sentences, sentence_vectors, labels):
    class_vectors = defaultdict(list)
    class_sentences = defaultdict(list)

    for sentence, vector, label in zip(sentences, sentence_vectors, labels):
        class_vectors[label].append(vector)
        class_sentences[label].append(sentence)

    classes = {}

    for label, vectors in class_vectors.items():
        vectors = np.asarray(vectors)
        center = np.mean(vectors, axis=0)
        dists = cdist(vectors, np.expand_dims(center, axis=0), metric="cosine")
        sorted_args = np.argsort(dists[:, 0])
        classes[label] = [
            (class_sentences[label][idx][0], dists[idx]) for idx in sorted_args
        ]

    classes = list(reversed(sorted(classes.values(), key=lambda item: len(item))))

    output = []

    for clas in classes:
        output.append([c[0] for c in clas])
        output.append([float(c[1]) for c in clas])

    return output


def save_results(sentences, sentence_vectors, labels, save_path):
    # PLOT CLUSTERS
    reduced_vectors = PCA(n_components=2).fit_transform(sentence_vectors)

    plt.figure()
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels)
    plt.xticks()
    plt.yticks()
    plt.savefig(os.path.join(save_path, "pca_clusters.png"))
    plt.close()

    reduced_vectors = TSNE(n_components=2).fit_transform(sentence_vectors)

    plt.figure()
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels)
    plt.xticks()
    plt.yticks()
    plt.savefig(os.path.join(save_path, "tsne_clusters.png"))
    plt.close()

    rows_data = get_rows(sentences, sentence_vectors, labels)

    is_domain_clustering = "domains" in save_path

    if is_domain_clustering:
        cluster_centers = [
            np.mean(sentence_vectors[labels == i], 0) for i in range(max(labels))
        ]

        with open(os.path.join(save_path, "clusters_centers.pck"), "wb") as f:
            pickle.dump(cluster_centers, f)

    for i in range(int(len(rows_data) / 2)):
        labels = rows_data[2 * i]
        dists = rows_data[2 * i + 1]

        plt.figure()
        plt.bar(range(len(labels)), dists)
        plt.ylabel("Cosine Distance")
        plt.title(f"Cluster_{i}")
        if is_domain_clustering:
            labels = [w[:10] for w in labels]
            plt.xticks(range(len(labels)), labels, rotation="vertical")
        else:
            plt.xticks([])
        plt.savefig(os.path.join(save_path, f"cluster_{i}_distances.png"))
        plt.close()

    # SAVE RESULTS TO CSV
    with open(os.path.join(save_path, "clusters.csv"), "w") as f:
        writer = csv.writer(f)

        n_rows = len(rows_data[0])

        for i in range(n_rows):
            row = []

            for column in rows_data:
                if i < len(column):
                    row.append(column[i])

            writer.writerow(row)


def sentence_vectorize(reduce_method, sents, vocab):
    bert_model = SentenceTransformer("distilbert-base-nli-mean-tokens", device="cpu")
    bert_sents = [" ".join(s[1]) for s in sents]
    if reduce_method == "sent":
        sentence_embeddings = bert_model.encode(bert_sents, show_progress_bar=False)

        return sentence_embeddings
    else:
        token_embeddings = bert_model.encode(
            bert_sents, output_value="token_embeddings", show_progress_bar=False,
        )

        def bert_mean_tokens(sent_embedding, weights=None):
            if weights:
                return np.average(
                    sent_embedding[1 : len(weights) + 1], 0, weights=weights
                )
            else:
                first_pad_idx = np.argmax(sent_embedding.sum(-1) == 0)
                return np.mean(sent_embedding[1 : first_pad_idx - 1], 0)

        if reduce_method == "mean":
            return [bert_mean_tokens(s) for s in token_embeddings]
        elif "icf_weight" in reduce_method:
            icf_weight = float(reduce_method.split("_")[-1])

            loaded_icf_terms = icf_terms()

            def process_sent(sent, sent_embeddings):
                _, sent = sent
                icf_words = []

                for i in range(len(sent)):
                    for r in range(1, len(sent) - i):
                        sub_term = " ".join(sent[i : i + r])
                        if sub_term in loaded_icf_terms[r]:
                            for word in sub_term.split():
                                icf_words.append(word)

                if len(icf_words) > 0:
                    icf_word_weight = icf_weight / len(icf_words)
                    other_word_weight = (1 - icf_weight) / (
                        len(sent) + 1 - len(icf_words)
                    )

                    tokenized_sent = bert_model._first_module().tokenizer.tokenize(
                        " ".join(sent)
                    )

                    weights = []
                    tokenized_idx = 0

                    for sent_idx, word in enumerate(sent):
                        if word == tokenized_sent[sent_idx + tokenized_idx]:
                            if word in icf_words:
                                weights.append(icf_word_weight)
                            else:
                                weights.append(other_word_weight)
                        else:
                            for i in range(
                                1, len(tokenized_sent) - sent_idx - tokenized_idx
                            ):
                                if (
                                    tokenized_sent[sent_idx + tokenized_idx + i][:2]
                                    != "##"
                                ):
                                    tokenized_idx += i - 1
                                    if word in icf_words:
                                        weights.extend([icf_word_weight / i] * i)
                                    else:
                                        weights.extend([other_word_weight / i] * i)
                                    break

                    return bert_mean_tokens(sent_embeddings, weights)
                else:
                    return bert_mean_tokens(sent_embeddings)

            return [
                process_sent(sent, sent_embs)
                for sent, sent_embs in zip(sents, token_embeddings)
            ]


def reduce_dim(sent_embeddings, reduced_dim):
    if "pca" in reduced_dim:
        return PCA(n_components=int(reduced_dim.split("_")[-1])).fit_transform(
            sent_embeddings
        )
    elif "tsne" in reduced_dim:
        return TSNE(
            n_components=int(reduced_dim.split("_")[-1]), init="pca"
        ).fit_transform(sent_embeddings)
    else:
        return sent_embeddings


@delayed
def launch_from_config(config, pre_config, base_path, vocab, sents, sent_embeddings):
    save_path = create_folder_for_config(config, pre_config, base_path)

    sent_embeddings = reduce_dim(sent_embeddings, config["reduced_dim"])

    score, labels = run_clustering(
        config["method"],
        config["clusters"],
        sents,
        sent_embeddings,
        base_path,
        config,
        pre_config,
        save_path,
    )

    save_results(sents, sent_embeddings, labels, save_path)

    return (config, score)


def get_hparams():
    hparams = OrderedDict()

    hparams["clusters"] = list(range(4, 8))
    hparams["reduced_dim"] = ["pca_2", "pca_5", "pca_10", "tsne_2", "tsne_5"]
    hparams["method"] = ["kmeans", "kmeans_icf_0.1", "kmeans_icf_0.5", "kmeans_icf_0.9"]

    pre_hparams = OrderedDict()

    pre_hparams["word_filtering"] = ["none"]
    pre_hparams["reduce_method"] = ["mean", "sent"]

    return hparams, pre_hparams


def domains_clustering():
    domains = load_all_domains()

    results_dir = "./results/domains"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results = OrderedDict()
    hparams, pre_hparams = get_hparams()

    all_configs = product(
        *[[(key, val) for val in vals] for key, vals in hparams.items()]
    )

    results = []

    for pre_config in tqdm(
        product(*[[(key, val) for val in vals] for key, vals in pre_hparams.items()]),
        desc="Processing configs for items",
    ):
        sents = deepcopy(domains)
        pre_config = dict(pre_config)
        vocab, sents = preprocess(
            sents, pre_config["word_filtering"], pre_config["reduce_method"]
        )

        sent_embeddings = sentence_vectorize(pre_config["reduce_method"], sents, vocab)

        results.extend(
            Parallel(n_jobs=-1)(
                launch_from_config(
                    dict(config), pre_config, results_dir, vocab, sents, sent_embeddings
                )
                for config in all_configs
            )
        )

    results = sorted(results, key=lambda item: item[1])

    for result in results:
        print(result)

    with open(os.path.join(results_dir, "results.pck"), "wb") as f:
        pickle.dump(results, f)


def items_clustering():
    items = load_all_items()

    results_dir = "./results/items"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results = OrderedDict()
    hparams, pre_hparams = get_hparams()

    results = []

    items_only_filters = [
        "automatic_filtering_10",
        "automatic_filtering_15",
        "automatic_filtering_20",
        "automatic_filtering_25",
    ]
    pre_hparams["word_filtering"] = pre_hparams["word_filtering"] + items_only_filters

    hparams["method"] = ["nearest_neighbor"] + hparams["method"]

    all_configs = product(
        *[[(key, val) for val in vals] for key, vals in hparams.items()]
    )

    for pre_config in tqdm(
        product(*[[(key, val) for val in vals] for key, vals in pre_hparams.items()]),
        desc="Processing configs for items",
    ):
        sents = deepcopy(items)
        pre_config = dict(pre_config)
        vocab, sents = preprocess(
            sents, pre_config["word_filtering"], pre_config["reduce_method"]
        )

        sent_embeddings = sentence_vectorize(pre_config["reduce_method"], sents, vocab)

        results.extend(
            Parallel(n_jobs=-1)(
                launch_from_config(
                    dict(config), pre_config, results_dir, vocab, sents, sent_embeddings
                )
                for config in all_configs
            )
        )

    results = sorted(results, key=lambda item: item[1])

    for result in results:
        print(result)

    with open(os.path.join(results_dir, "results.pck"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    domains_clustering()
    items_clustering()
