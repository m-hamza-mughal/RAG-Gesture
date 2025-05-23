import torch
import numpy as np
import librosa
import copy
# import gensim.downloader as api
from fuzzywuzzy import fuzz
# word2vec_model = api.load("word2vec-google-news-300")
# fasttext_model = api.load("fasttext-wiki-news-subwords-300")


class TextFeatureExtractor:
    """
    Text feature extractor class
    """

    def __init__(self, text_encoder, text_tokenizer):
        """
        Initialize the text feature extractor

        Args:
            text_encoder (transformers.PreTrainedModel): text encoder model
            text_tokenizer (transformers.PreTrainedTokenizer): text tokenizer
        """
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.features_cache = {}
        self.device = self.text_encoder.device

    def extract_feature(self, text, normalize=True):
        """
        Extract text features from the text encoder

        Args:
            text (str): input text
        """
        # make sure text contains all unicode characters
        text = text.encode("utf-8", "ignore").decode("utf-8")

        # check if text features are already computed
        if hash(text) in self.features_cache:
            text_features = self.features_cache[hash(text)].to(self.device)
        else:
            text_tokens = self.text_tokenizer(text, return_tensors="pt").to(self.device)
            text_features = self.text_encoder(**text_tokens).last_hidden_state

            # get first element of the output (because of the batch size of 1)

            text_features = text_features[0]

            # cache the text features
            self.features_cache[hash(text)] = text_features.detach().cpu()

        # normalize the last dimension
        if normalize:
            text_features = text_features / torch.norm(
                text_features, dim=-1, keepdim=True
            )
        return text_features


def extract_audio_f0(audio, start, end, sr=16000):
    """
    Extract FO from the audio signal

    Args:
        audio (np.ndarray): audio signal
        start (float): start time in seconds
        end (float): end time in seconds
    """
    audio_mask = np.zeros_like(audio)
    start = int(start * sr)
    end = int(end * sr)
    audio_mask[start:end] = 1

    audio_masked = audio * audio_mask

    f0, voicing, voicing_prob = librosa.pyin(
        y=audio_masked,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    return f0


def sort_sidx_by_textsimilarity(
    sorted_smp_indexes, text, encoded_text, feature_cache
):
    """
    Sort the sample indexes based on text similarity

    Args:
        sorted_smp_indexes: sorted sample indexes
        text: text in query
        feature_extractor: text feature extractor
        feature_cache: text feature cache
    """
    if len(sorted_smp_indexes) == 0:
        return sorted_smp_indexes

    # text_feature = feature_extractor.extract_feature(text)
    text_feature = encoded_text
    # breakpoint() # check the text feature passed from the dataloader
    # text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)

    # breakpoint()

    dbidx_2_simscore = {}
    for smp_idx in sorted_smp_indexes:

        text_f_norm, s_id = feature_cache[smp_idx]
        text_f_norm = text_f_norm.to(text_feature.device)
        # text_f_norm = text_f[0] / text_f[0].norm(dim=1, keepdim=True)

        text_sim_matrix = torch.mm(text_feature, text_f_norm.T)
        # get the diagonal elements
        text_diag = torch.diagonal(text_sim_matrix)
        text_sim = text_diag.mean()

        # multiply the off diagonal elements by 0.5
        # text_sim_matrix = torch.eye(**text_sim_matrix.shape) * text_sim_matrix + 0.5 * (1 - torch.eye(**text_sim_matrix.shape)) * text_sim_matrix
        # text_sim = text_sim_matrix.mean()

        dbidx_2_simscore[smp_idx] = text_sim

    # breakpoint()
    sorted_smp_indexes = sorted(
        dbidx_2_simscore, key=dbidx_2_simscore.get, reverse=True
    )  # these are sample indexes
    # print('No disco Time taken:', time.time() - start)

    return sorted_smp_indexes


def sort_sidx_by_textsimilarity_batched(
         text, encoded_text, feature_cache_tensor, sample_names
):
    """
    Sort the sample indexes based on text similarity

    Args:
        sorted_smp_indexes: sorted sample indexes
        text: text in query
        feature_extractor: text feature extractor
        feature_cache: text feature cache
    """

    # text_feature = feature_extractor.extract_feature(text)
    text_feature = encoded_text
    # breakpoint() # check the text feature passed from the dataloader
    # text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)

    # breakpoint()
      # shape: (N, D)
    feature_cache_tensor = feature_cache_tensor.to(text_feature.device)
    text_feature_pad = torch.nn.functional.pad(text_feature, (0, 0, 0, feature_cache_tensor.shape[1] - text_feature.shape[0]), value=0)
    text_feature_tensor = text_feature_pad.unsqueeze(0).expand(feature_cache_tensor.shape[0], -1, -1)
    

    text_sim_matrix = torch.bmm(text_feature_tensor, feature_cache_tensor.transpose(1, 2))
    text_diag = torch.diagonal(text_sim_matrix, dim1=-2, dim2=-1)
    text_sim = text_diag.mean(dim=-1)

    sim_values, sim_indices = text_sim.sort(descending=True, stable=True)

    sorted_smp_indexes = [sample_names[i.item()] for i in sim_indices]

    return sorted_smp_indexes


def map_conns_to_prominence(conn_list, prominence_list):
    """
    this function maps connective list (smaller) to prominence list (larger)

    Args:
        conn_list: list of connectives
        prominence_list: list of prominences
    """
    relevant_dps = {}
    residual_conns = copy.deepcopy(conn_list)
    # print(prominence_list)
    for dp_idx, dp in enumerate(prominence_list):
        dp_word = dp[0]  # .lower()

        # only keep alphanumeric characters and space in dp_word (remove special characters)
        dp_word = "".join([c for c in str(dp_word) if c.isalnum() or c.isspace()])
        # dp_word = dp_word.lower()

        for si, sc in enumerate(conn_list):
            if si not in relevant_dps:
                relevant_dps[si] = []

            if residual_conns[si] is None:
                continue

            # only keep alphanumeric characters and space in sc (remove special characters)
            sc = "".join([c for c in sc if c.isalnum() or c.isspace()])
            # sc = sc.lower()

            if dp_word == sc or dp_word in sc.split():

                prom_value = dp[3]
                relevant_dps[si].append((sc, prom_value))

                if dp_word == sc or dp_word == sc.split()[-1]:
                    residual_conns[si] = None
                break

    # merge the elements of relevant_dps[si] if len(relevant_dps[si]) > 1 
    # i.e. connective has multiple words
    for si, dps in relevant_dps.items():
        if len(dps) > 1:
            sc_temp = "".join([c for c in conn_list[si] if c.isalnum() or c.isspace()])
            if dps[0][0] != sc_temp:
                breakpoint()
            assert dps[0][0] == sc_temp
            relevant_dps[si] = (
                conn_list[si],
                sum([d[1] for d in dps]) / len(dps),
            )
        else:
            relevant_dps[si] = dps[0] if len(dps) > 0 else None

    # print(relevant_dps, "\n", conn_list, "\n****")
    if len(relevant_dps) != len(conn_list):
        breakpoint()
    assert len(relevant_dps) == len(conn_list)  # same length as conn_list
    return relevant_dps


def run_similarity_model(word1, word2):
    "run either word2vec or fasttext model"
    try:
        return word2vec_model.similarity(word1, word2)
    except KeyError:
        return fasttext_model.similarity(word1, word2)


def get_word_similarity_score(word1, word2):
    """
    Get word similarity score based on word2vec/fasttext embeddings

    Args:
        word1: word 1
        word2: word 2
    """
    try:
        if len(word1.split()) > 1 and len(word2.split()) == 1:
            avg_sim = 0
            for w in word1.split():
                avg_sim += run_similarity_model(w, word2)

            return avg_sim / len(word1.split())
        elif len(word2.split()) > 1 and len(word1.split()) == 1:
            avg_sim = 0
            for w in word2.split():
                avg_sim += run_similarity_model(word1, w)

            return avg_sim / len(word2.split())
        elif len(word1.split()) > 1 and len(word2.split()) > 1:
            avg_sim = 0
            for w1 in word1.split():
                for w2 in word2.split():
                    avg_sim += run_similarity_model(w1, w2)

            return avg_sim / (len(word1.split()) * len(word2.split()))
        else:
            return run_similarity_model(word1, word2)
    except Exception as e:
        return fuzz.partial_ratio(word1, word2) / 100
        # print("word similarity error:", e)
        # return 0.0
