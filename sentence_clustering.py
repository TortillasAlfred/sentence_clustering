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


def create_folder_for_config(config, base_path):
    save_path = base_path

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
        filtering = lambda _: True
        sents = [list(filter(filtering, sent)) for sent in sents]
        sents = [(raw_sents[i], sent) for i, sent in enumerate(sents) if len(sent) > 0]
    elif word_filtering == "stopwords":
        from nltk.corpus import stopwords

        sents = [sent.split() for sent in sents]
        stop_words = set(stopwords.words("english"))
        filtering = lambda word: word not in stop_words
        sents = [list(filter(filtering, sent)) for sent in sents]
        sents = [(raw_sents[i], sent) for i, sent in enumerate(sents) if len(sent) > 0]
    elif word_filtering == "len3":
        filtering = lambda word: len(word) > 3
        sents = [sent.split() for sent in sents]
        sents = [list(filter(filtering, sent)) for sent in sents]
        sents = [(raw_sents[i], sent) for i, sent in enumerate(sents) if len(sent) > 0]
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

    if isinstance(vectors, int):
        vocab = Vocab(
            vocab_counter,
            vectors=f"glove.6B.{vectors}d",
            vectors_cache="/home/magod/scratch/embeddings/",
            specials=[],
        )
        vocab.vectors = vocab.vectors.numpy()
    else:
        vocab = Vocab(vocab_counter, vectors=None, specials=[],)

    return vocab, sents


def get_clustering_obj(method, clusters):
    if method == "kmeans":
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
    else:
        raise ValueError("Unknown clustering method")


def run_clustering(method, clusters, sentences, sentence_vectors, base_path):
    clustering_obj = get_clustering_obj(method, clusters)

    labels = clustering_obj.fit_predict(sentence_vectors)

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
        classes[label] = [class_sentences[label][idx][0] for idx in sorted_args]

    classes = list(reversed(sorted(classes.values(), key=lambda item: len(item))))

    return classes


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

    # SAVE RESULTS TO CSV
    with open(os.path.join(save_path, "clusters.csv"), "w") as f:
        writer = csv.writer(f)

        rows_data = get_rows(sentences, sentence_vectors, labels)
        n_rows = len(rows_data[0])

        for i in range(n_rows):
            row = []

            for column in rows_data:
                if i < len(column):
                    row.append(column[i])

            writer.writerow(row)


def sentence_vectorize(vector_method, sents, vocab):
    if vector_method == "bert_sent":
        bert_model = SentenceTransformer("bert-large-nli-mean-tokens", device="cpu")
        sentence_embeddings = bert_model.encode([s[0] for s in sents])
        return sentence_embeddings
    elif vector_method == "bert_sent_pca":
        bert_model = SentenceTransformer("bert-large-nli-mean-tokens", device="cpu")
        sentence_embeddings = bert_model.encode([s[0] for s in sents])
        return sentence2vec(sents, vocab, sent_embeddings=sentence_embeddings)
    elif vector_method == "bert_word":
        bert_model = SentenceTransformer("bert-large-nli-mean-tokens", device="cpu")
        token_embeddings = bert_model.encode(
            [s[0] for s in sents], output_value="token_embeddings"
        )
        return sentence2vec(sents, vocab, token_embeddings=token_embeddings)
    elif vector_method == "custom":
        custom_model = fasttext.load_model(
            "/scratch/magod/mobility_abstracts/fasttext_model.bin"
        )
        token_embeddings = [
            [custom_model.get_word_vector(w) for w in s[1]] for s in sents
        ]
        return sentence2vec(sents, vocab, token_embeddings=token_embeddings)
    else:
        return sentence2vec(sents, vocab)


def apply_pca(sent_embeddings, pca_dim):
    if pca_dim:
        return PCA(n_components=pca_dim).fit_transform(sent_embeddings)
    else:
        return sent_embeddings


@delayed
def launch_from_config(config, base_path, sents):
    save_path = create_folder_for_config(config, base_path)

    vocab, sents = preprocess(sents, config["word_filtering"], config["vectors"])
    sent_embeddings = sentence_vectorize(config["vectors"], sents, vocab)
    sent_embeddings = apply_pca(sent_embeddings, config["pca_dim"])

    score, labels = run_clustering(
        config["method"], config["clusters"], sents, sent_embeddings, base_path
    )

    save_results(sents, sent_embeddings, labels, save_path)

    return (config, score)


def get_hparams():
    hparams = OrderedDict()

    hparams["clusters"] = list(range(4, 9))
    hparams["word_filtering"] = ["none", "stopwords", "len3"]
    hparams["vectors"] = [
        "bert_sent_pca",
        "custom",
        50,
        "bert_word",
        100,
        "bert_sent",
        200,
        300,
    ]
    hparams["pca_dim"] = [2, 5, 10, 25, None]
    hparams["method"] = ["kmeans"]

    return hparams


def domains_clustering():
    domains = load_all_domains()

    results_dir = "./results/domains"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results = OrderedDict()
    hparams = get_hparams()

    all_configs = product(
        *[[(key, val) for val in vals] for key, vals in hparams.items()]
    )
    all_configs = tqdm(
        [OrderedDict(config) for config in all_configs],
        desc="Processing configs for domains",
    )

    results = Parallel(n_jobs=-1, verbose=1)(
        launch_from_config(config, results_dir, domains) for config in all_configs
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
    hparams = get_hparams()
    hparams["word_filtering"] = ["word_groups"] + hparams["word_filtering"]

    all_configs = product(
        *[[(key, val) for val in vals] for key, vals in hparams.items()]
    )
    all_configs = tqdm(
        [OrderedDict(config) for config in all_configs],
        desc="Processing configs for items",
    )

    results = Parallel(n_jobs=-1, verbose=1)(
        launch_from_config(config, results_dir, items) for config in all_configs
    )

    results = sorted(results, key=lambda item: item[1])

    for result in results:
        print(result)

    with open(os.path.join(results_dir, "results.pck"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    items_clustering()
    domains_clustering()
