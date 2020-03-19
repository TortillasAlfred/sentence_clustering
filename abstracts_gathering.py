from utils import configure_logging

import argparse
import logging
import os
import pickle
import unicodedata
import nltk

from tqdm import tqdm
from Bio import Entrez

N_IDS_PER_READ_REQUEST = 10000000
N_IDS_PER_FETCH_REQUEST = 100


def collect_mesh(mesh_term, ids_dir):
    output_path = os.path.join(ids_dir, f"{mesh_term}.pck")

    if os.path.isfile(output_path):
        return

    handle = Entrez.esearch(db="pubmed", retmax=1, term=mesh_term, rettype="json")

    record = Entrez.read(handle)

    n_requests_needed = int(int(record["Count"]) / N_IDS_PER_REQUEST) + 1

    mesh_ids = []

    for i in tqdm(range(n_requests_needed)):
        handle = Entrez.esearch(
            db="pubmed",
            retmax=N_IDS_PER_REQUEST,
            retstart=i * N_IDS_PER_REQUEST,
            term=mesh_term,
            rettype="json",
        )

        mesh_ids.extend(Entrez.read(handle)["IdList"])

    mesh_ids = list(map(str, mesh_ids))

    with open(output_path, "wb") as f:
        pickle.dump(mesh_ids, f, protocol=pickle.HIGHEST_PROTOCOL)


def regroup_ids(mesh_terms, ids_dir):
    all_ids_dir = os.path.join(ids_dir, "all_ids.pck")

    if os.path.isfile(all_ids_dir):
        return

    all_ids = []

    for mesh_term in mesh_terms:
        with open(os.path.join(ids_dir, f"{mesh_term}.pck"), "rb") as f:
            all_ids.extend(pickle.load(f))

    logging.info(f"A total of {len(all_ids)} ids were collected.")

    all_ids = list(set(all_ids))

    logging.info(f"After removing duplicates, {len(all_ids)} were left.")

    logging.info(f"Writing all ids to {all_ids_dir}.")

    with open(all_ids_dir, "wb") as f:
        pickle.dump(all_ids, f)


def read_abstracts(texts_dir, ids_dir):
    all_ids_path = os.path.join(ids_dir, "all_ids.pck")

    current_slice_path = os.path.join(texts_dir, "current_slice.pck")

    if os.path.isfile(current_slice_path):
        with open(current_slice_path, "rb") as f:
            current_slice = pickle.load(f)
    else:
        current_slice = 0

    with open(all_ids_path, "rb") as f:
        all_ids = pickle.load(f)

    os.makedirs(texts_dir, exist_ok=True)

    n_requests = int(len(all_ids) / N_IDS_PER_REQUEST) + 1

    for i in tqdm(list(range(current_slice, n_requests)), desc="Reading summaries..."):
        with open(current_slice_path, "wb") as f:
            pickle.dump(i, f)

        begin_idx = i * N_IDS_PER_REQUEST
        end_idx = begin_idx + N_IDS_PER_REQUEST
        ids = all_ids[begin_idx:end_idx]

        read_abstracts_ids(texts_dir, ids)


def read_abstracts_ids(texts_dir, ids):
    handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")

    articles = Entrez.read(handle)["PubmedArticle"]

    logging.info("Reading done.")

    for article, id in tqdm(
        list(zip(articles, ids)), desc="Preprocessing read articles..."
    ):
        output_path = os.path.join(texts_dir, f"{id}.txt")

        if os.path.isfile(output_path):
            continue

        if "Abstract" not in article["MedlineCitation"]["Article"]:
            continue

        article_gists = article["MedlineCitation"]["Article"]["Abstract"][
            "AbstractText"
        ]
        abstract = []

        for gist in article_gists:
            gist = str(gist).lower()
            gist = unicodedata.normalize("NFKD", gist)
            sents = []
            for sent in nltk.sent_tokenize(gist):
                sents.append(" ".join(nltk.word_tokenize(sent)))
            abstract.extend(sents)

        abstract = "\n".join(abstract)

        with open(output_path, "w") as f:
            f.write(abstract)


def preprocess_data(data):
    abstracts = []

    for article in data:
        if "Abstract" not in article["MedlineCitation"]["Article"]:
            continue

        article_gists = article["MedlineCitation"]["Article"]["Abstract"][
            "AbstractText"
        ]

        for gist in article_gists:
            gist = str(gist).lower()
            gist = unicodedata.normalize("NFKD", gist)
            sents = []
            for sent in nltk.sent_tokenize(gist):
                # sents.append(" ".join(nltk.word_tokenize(sent)))
                sents.append(sent)  # No tokenization yet
            abstracts.extend(sents)

    abstracts = "\n".join(abstracts)

    return abstracts


def process_mesh_term(mesh_term, texts_dir):
    output_path = os.path.join(texts_dir, f"{mesh_term}.txt")

    handle = Entrez.esearch(
        db="pubmed",
        term=mesh_term,
        usehistory="y",
        retmax=N_IDS_PER_READ_REQUEST,
        rettype="json",
    )
    search_results = Entrez.read(handle)
    handle.close()

    count = int(search_results["Count"])
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]

    with open(output_path, "w") as output_file:
        for start in tqdm(
            list(range(0, count, N_IDS_PER_FETCH_REQUEST)), desc="Processing slices ..."
        ):
            end = min(count, start + N_IDS_PER_FETCH_REQUEST)
            fetch_handle = Entrez.efetch(
                db="pubmed",
                retmode="xml",
                retstart=start,
                retmax=N_IDS_PER_FETCH_REQUEST,
                webenv=webenv,
                query_key=query_key,
            )
            data = Entrez.read(fetch_handle)["PubmedArticle"]
            fetch_handle.close()
            data = preprocess_data(data)

            output_file.write(data)


def main(options):
    configure_logging()

    Entrez.email = options.email

    logging.info(
        f"Beginning abstracts gathering for mesh terms located in file {options.mesh_terms_path}"
    )

    with open(options.mesh_terms_path, "r") as mesh_file:
        mesh_terms = mesh_file.read().splitlines()

    logging.info(f"The following mesh terms were retrieved {', '.join(mesh_terms)}")

    os.makedirs(options.texts_dir, exist_ok=True)

    for mesh_term in tqdm(mesh_terms, desc="Processing mesh terms..."):
        process_mesh_term(mesh_term, options.texts_dir)

    logging.info("Done")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--mesh_terms_path", type=str, default="./mesh_terms.txt"
    )
    argument_parser.add_argument("--texts_dir", type=str, default="./mesh_texts/")
    argument_parser.add_argument(
        "--email", type=str, default="mathieu.godbout.3@ulaval.ca"
    )
    options = argument_parser.parse_args()
    main(options)
