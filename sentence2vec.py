#!/usr/bin/python3

#
#  Copyright 2016-2018 Peter de Vocht
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from typing import List

# see spacy_sentence2vec.py for an example usage with real language inputs


# an embedding word with associated vector
class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

    def __str__(self):
        return self.text + " : " + str(self.vector)

    def __repr__(self):
        return self.__str__()


# a sentence, a list of words
class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    # return the length of a sentence
    def len(self) -> int:
        return len(self.word_list)

    def __str__(self):
        word_str_list = [word.text for word in self.word_list]
        return " ".join(word_str_list)

    def __repr__(self):
        return self.__str__()


# todo: get a proper word frequency for a word in a document set
# or perhaps just a typical frequency for a word from Google's n-grams
def get_word_frequency(word_text):
    return 0.0001  # set to a low occurring frequency - probably not unrealistic for most words, improves vector values


# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# Sanjeev Arora, Yingyu Liang, Tengyu Ma
# Princeton University
# convert a list of sentence with word2vec items into a set of sentence vectors
def sentence_to_vec(
    sentence_list: List[Sentence], embedding_size: int, vocab, a: float = 1e-3,
):
    num_tokens = sum(vocab.freqs.values())

    freqs = defaultdict(lambda _: 1)
    freqs.update(vocab.freqs)

    sentence_set = []
    for sentence in sentence_list:
        # add all word2vec values into one vector for the sentence
        vs = np.zeros(embedding_size)
        sentence_length = sentence.len()
        for word in sentence.word_list:
            # smooth inverse frequency, SIF
            a_value = a / (a + freqs[word.text] / num_tokens)
            # vs += sif * word_vector
            vs = np.add(vs, np.multiply(a_value, word.vector))

        vs = np.divide(vs, sentence_length)  # weighted average
        # add to our existing re-calculated set of sentences
        sentence_set.append(vs)

    # calculate PCA of this sentence set
    pca = PCA()
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            # add needed extension for multiplication below
            u = np.append(u, 0)

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return sentence_vecs


def sentence2vec(sentences, vocab, sent_embeddings=None, token_embeddings=None):
    if sent_embeddings:
        sent_embeddings = np.array(sent_embeddings)
        embedding_size = sent_embeddings.shape[1]

        pca = PCA()
        pca.fit(np.array(sent_embeddings))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT

        # pad the vector?  (occurs if we have less sentences than embeddings_size)
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                # add needed extension for multiplication below
                u = np.append(u, 0)

        # resulting sentence vectors, vs = vs -u x uT x vs
        sentence_vectors = []
        for sent in sent_embeddings:
            sub = np.multiply(u, sent)
            sentence_vectors.append(np.subtract(sent, sub))
    elif token_embeddings:
        token_embeddings = np.array(token_embeddings)

        # convert the above sentences to vectors using spacy's large model vectors
        sentence_list = []
        for sentence, sent_tokens in zip(sentences, token_embeddings):
            word_list = []
            for word, emb in zip(sentence[1], sent_tokens):
                word_list.append(Word(word, emb))
            if len(word_list) > 0:  # did we find any words (not an empty set)
                sentence_list.append(Sentence(word_list))

        # apply single sentence word embedding
        embedding_size = token_embeddings[0].shape[-1]

        sentence_vectors = sentence_to_vec(
            sentence_list, embedding_size, vocab
        )  # all vectors converted together
    else:
        # convert the above sentences to vectors using spacy's large model vectors
        sentence_list = []
        for sentence in sentences:
            word_list = []
            for word in sentence[1]:
                word_list.append(Word(word, vocab.vectors[vocab.stoi[word]]))
            if len(word_list) > 0:  # did we find any words (not an empty set)
                sentence_list.append(Sentence(word_list))

        # apply single sentence word embedding
        embedding_size = vocab.vectors.shape[1]

        sentence_vectors = sentence_to_vec(
            sentence_list, embedding_size, vocab
        )  # all vectors converted together

    return sentence_vectors
