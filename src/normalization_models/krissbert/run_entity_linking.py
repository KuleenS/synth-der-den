# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Run entity linking
"""
import argparse
import os
import glob
import pathlib
import pickle
import time
import math
import multiprocessing
import tomli
from typing import List, Tuple, Dict, Iterator, Set
from functools import partial
from multiprocessing.dummy import Pool

import mysql.connector as mariadb

import pandas as pd

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
import faiss

from tqdm import tqdm

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
)

from src.normalization_models.krissbert.data.semeval.semeval_data import SemevalMentionsDataset
from src.normalization_models.krissbert.data.bc5dr.bc5dr_data import BCD5DRDataset
from src.normalization_models.krissbert.data.ncbi.ncbi_data import NCBIDataset

from src.normalization_models.krissbert.utils import generate_vectors

class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(
        self, query_vectors: np.array, top_docs: int
    ) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".index.dpr"
            meta_file = path + ".index_meta.dpr"
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)


    def init_index(self, vector_sz: int):
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in tqdm(range(0, n, self.buffer_size), desc="index"):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [
                np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]
            ]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)

        indexed_cnt = len(self.index_id_to_db_id)

    def search_knn(
        self, query_vectors: np.array, top_docs: int, batch_size: int = 4096,
    ) -> List[Tuple[List[object], List[float]]]:
        num_queries = query_vectors.shape[0]
        scores, indexes = [], []
        for start in tqdm(range(0, num_queries, batch_size), desc="Search"):
            batch_vectors = query_vectors[start:start + batch_size]
            batch_scores, batch_indexes = self.index.search(batch_vectors, top_docs)
            scores.extend(batch_scores)
            indexes.extend(batch_indexes)
        # convert to external ids
        
        db_ids = [
            [self.index_id_to_db_id[i] for i in query_top_idxs if i != -1]
            for query_top_idxs in indexes
        ]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def get_index_name(self):
        return "flat_index"


def load_umls_data(files_patterns: List[str], candidate_ids: Dict = None) -> Dict:
    input_paths = []
    for pattern in files_patterns:
        pattern_files = glob.glob(pattern)
        input_paths.extend(pattern_files)
    umls_data = {}
    for file in sorted(input_paths):
        with open(file, "rb") as reader:
            for meta, vec in pickle.load(reader):
                assert len(meta['cuis']) == 1, breakpoint()
                cui = meta['cuis'][0]
                if candidate_ids and cui not in candidate_ids:
                    continue
                umls_data[cui] = (meta, vec)
    return umls_data


def iterate_encoded_files(
    vector_files: list,
    candidate_ids: Set = None,
    umls_data: Dict = None,
)-> Iterator:
    proto_data = {}
    for file in vector_files:
        with open(file, "rb") as reader:
            for meta, vec in pickle.load(reader):
                cuis = meta['cuis']
                if candidate_ids and all(c not in candidate_ids for c in cuis):
                    continue
                for cui in cuis:
                    proto_data.setdefault(cui, []).append((meta, vec))
    # Concatenate prototype embs with additional knowledge embs from UMLS.
    if umls_data is not None:
        for cui, (meta, vec) in umls_data.items():
            if cui in proto_data:
                for _, _vec in proto_data.pop(cui):
                    extended_vec = np.concatenate((vec, _vec), axis=0)
                    yield (meta, extended_vec)
            else:
                extended_vec = np.concatenate((vec, np.zeros_like(vec)), axis=0)
                yield (meta, extended_vec)
    for cui in list(proto_data.keys()):
        for meta, vec in proto_data.pop(cui):
            extended_vec = np.concatenate((np.zeros_like(vec), vec), axis=0)
            yield (meta, extended_vec)
    assert len(proto_data) == 0


class DenseRetriever:
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
    ):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def generate_mention_vectors(self, ds: torch.utils.data.Dataset) -> T:
        self.encoder.eval()
        return generate_vectors(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            dataset=ds,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )


class FaissRetriever(DenseRetriever):
    """
    Does entity retrieving over the provided index and encoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        index: DenseIndexer,
    ):
        super().__init__(encoder, tokenizer, batch_size, max_length)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        candidate_ids: Set = None,
        umls_data: Dict = None,
    ):
        """
        Indexes encoded data takes form a list of files
        :param vector_files: a list of files
        :param buffer_size: size of a buffer to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(
            iterate_encoded_files(vector_files, candidate_ids, umls_data)
        ):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)

    def get_top_hits(
        self, mention_vectors: np.array, top_k: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching given the mention vectors batch
        """
        time0 = time.time()
        search = partial(
            self.index.search_knn,
            top_docs=top_k,
        )
        results = []
        num_processes = multiprocessing.cpu_count()
        shard_size = math.ceil(mention_vectors.shape[0] / num_processes)
        shards = []
        for i in range(0, mention_vectors.shape[0], shard_size):
            shards.append(mention_vectors[i:i + shard_size])
        with Pool(processes=num_processes) as pool:
            it = pool.map(search, shards)
            for ret in it:
                results += ret
            # results = self.index.search_knn(mention_vectors, top_k)
        self.index = None
        return results


def hit(pred: List[str], gold: List[str]) -> bool:
    return all(p in gold for p in pred)


def dedup_ids(ids: List[Dict]) -> List[Dict]:
    deduped_ids = []
    seen_cuis = set()
    for d in ids:
        if all(cui in seen_cuis for cui in d['cuis']):
            continue
        seen_cuis.update(d['cuis'])
        deduped_ids.append(d)
    return deduped_ids


def evaluate(
    ds: torch.utils.data.Dataset,
    result_ent_ids: List[Tuple[List[object], List[float]]],
    lookup_table: str,
    top_ks: List[int] = (1, 5, 10, 50, 100),
) -> List[Dict]:
    lut = {}
    with open(lookup_table, encoding='utf-8') as f:
        for ln in f:
            cuis, name = ln.strip().split('||')
            cuis = cuis.split('|')
            lut[name] = cuis

    n = len(ds)
    top_k_hits = {top_k: 0 for top_k in top_ks}
    for i in range(len(result_ent_ids)):
        d = ds[i]
        ids, _ = result_ent_ids[i]
        ids = dedup_ids(ids)
        ids = ids[:max(top_ks)]
        candidates = [
            {'cuis': eid['cuis'], 'hit': int(hit(pred=eid['cuis'], gold=d.cuis))}
            for eid in ids
        ]
        lut_cuis = lut.get(d.mention, [])
        if len(lut_cuis) == 1:
            # If the mention only has one ID in the look up table,
            # we use the ID as the top prediction.
            candidates.insert(
                0,
                {'cuis': lut_cuis, 'hit': int(hit(pred=lut_cuis, gold=d.cuis))}
            )
        for top_k in top_k_hits:
            if any(c['hit'] for c in candidates[:top_k]):
                top_k_hits[top_k] += 1

    top_k_acc = {top_k: v / n for top_k, v in top_k_hits.items()}
    print("Top-k accuracy %s", top_k_acc)


def main(args):

    config_path = args.config

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    set_seed(config["seed"])

    # Load pretrained.
    bert_config = AutoConfig.from_pretrained(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        use_fast=True,
    )
    encoder = AutoModel.from_pretrained(
        config["model"],
        config=bert_config
    )
    encoder.cuda()
    encoder.eval()
    
    vector_size = bert_config.hidden_size

    user = os.environ["UMLS_USER"]
    pwd = os.environ["UMLS_PWD"]
    database = os.environ["UMLS_DATABASE_NAME"]
    ip = os.environ["UMLS_IP"]

    mariadb_connection = mariadb.connect(
        user=user, password=pwd, database=database, host=ip
    )

    df_omim = pd.read_sql(
        "SELECT CUI, SAB, CODE, STR FROM MRCONSO WHERE SAB LIKE '%OMIM%'",
        con=mariadb_connection,
    )
    df_msh = pd.read_sql(
        "SELECT CUI, SAB, CODE, STR FROM MRCONSO WHERE SAB LIKE '%MSH%'",
        con=mariadb_connection,
    )

    df_omim.columns = ["CUI", "OMIM_SAB", "OMIM_CODE", "OMIM_STR"]
    df_msh.columns = ["CUI", "MSH_SAB", "MSH_CODE", "MSH_STR"]

    omim_to_cui = dict(zip(list(df_omim["OMIM_CODE"]), list(df_omim["CUI"])))

    mesh_to_cui = dict(zip(list(df_msh["MSH_CODE"]), list(df_msh["CUI"])))

    print("Regular Accuracy")

    dataset = config["dataset"]

    # Load test data.
    if dataset == "semeval":
        ds = SemevalMentionsDataset(config["test_data"])
    elif dataset == "ncbi":
        ds = NCBIDataset("test", ood=False, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)
    elif dataset == "bc5dr":
        ds = BCD5DRDataset("test", ood=False, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)

    # Init indexer.
    index = DenseFlatIndexer()
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size * 2)

    # candidate ids
    candidate_ids = None
    if "entity_list_ids" in config:
        with open(config["entity_list_ids"], encoding='utf-8') as f:
            candidate_ids = set(f.read().split('\n'))

    # Start indexing
    input_paths = []
    for pattern in config["encoded_files"]:
        pattern_files = glob.glob(pattern)
        input_paths.extend(pattern_files)
    input_paths = sorted(set(input_paths))

    retriever = FaissRetriever(
        encoder, tokenizer, config["batch_size"],config["max_length"], index)
    mentions_tensor = retriever.generate_mention_vectors(ds)

    # Load UMLS knowledge
    umls_data = None
    if "encoded_umls_files" in config:
        umls_data = load_umls_data(config["encoded_umls_files"], candidate_ids)

    index_path = config.get("index_path", None)
    if index_path and index.index_exists(index_path):
        retriever.index.deserialize(index_path)
    else:
        retriever.index_encoded_data(
            vector_files=input_paths,
            buffer_size=index_buffer_sz,
            candidate_ids=candidate_ids,
            umls_data=umls_data,
        )
        if index_path:
            pathlib.Path(os.path.dirname(index_path)).mkdir(
                parents=True, exist_ok=True)
            retriever.index.serialize(index_path)

    # Encode test data.
    mentions_tensor = torch.cat([mentions_tensor, mentions_tensor], dim=1)

    # To get k different entities, we retrieve 32 * k mentions and then dedup.
    top_ids_and_scores = retriever.get_top_hits(
        mentions_tensor.numpy(), config["num_retrievals"] * 32)

    evaluate(ds, top_ids_and_scores, config["entity_list_names"])

    print("OOD Accuracy")

    if dataset == "semeval":
        ds = SemevalMentionsDataset(config["test_data"], ood=True)
    elif dataset == "ncbi":
        ds = NCBIDataset("test", ood=True, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)
    elif dataset == "bc5dr":
        ds = BCD5DRDataset("test", ood=True, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)

    # Init indexer.
    index = DenseFlatIndexer()
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size * 2)

    # candidate ids
    candidate_ids = None
    if "entity_list_ids" in config:
        with open(config["entity_list_ids"], encoding='utf-8') as f:
            candidate_ids = set(f.read().split('\n'))

    # Start indexing
    input_paths = []
    for pattern in config["encoded_files"]:
        pattern_files = glob.glob(pattern)
        input_paths.extend(pattern_files)
    input_paths = sorted(set(input_paths))

    retriever = FaissRetriever(
        encoder, tokenizer, config["batch_size"],config["max_length"], index)
    mentions_tensor = retriever.generate_mention_vectors(ds)

    # Load UMLS knowledge
    umls_data = None
    if "encoded_umls_files" in config:
        umls_data = load_umls_data(config["encoded_umls_files"], candidate_ids)

    index_path = config.get("index_path", None)
    if index_path and index.index_exists(index_path):
        retriever.index.deserialize(index_path)
    else:
        retriever.index_encoded_data(
            vector_files=input_paths,
            buffer_size=index_buffer_sz,
            candidate_ids=candidate_ids,
            umls_data=umls_data,
        )
        if index_path:
            pathlib.Path(os.path.dirname(index_path)).mkdir(
                parents=True, exist_ok=True)
            retriever.index.serialize(index_path)

    # Encode test data.
    mentions_tensor = torch.cat([mentions_tensor, mentions_tensor], dim=1)

    # To get k different entities, we retrieve 32 * k mentions and then dedup.
    top_ids_and_scores = retriever.get_top_hits(
        mentions_tensor.numpy(), config["num_retrievals"] * 32)

    evaluate(ds, top_ids_and_scores, config["entity_list_names"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config")


    args = parser.parse_args()

    main(args)


