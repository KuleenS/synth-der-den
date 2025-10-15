import os
import copy
from pathlib import Path
from typing import List, Dict, Any, Tuple
from random import sample

from datasets import load_dataset

from intervaltree import IntervalTree

import spacy

import polars as pl

import numpy as np

import sqlalchemy as sa

import pandas as pd

from tqdm import tqdm

CUI_FIX = {"C1883552":"C0004093",
"C0021311":"C0009450",
"C0342243":"C0020456",
"C0242362":"C0021818",
"C0228070":"C0027757",
"C1704437":"C0035220",
"C2919362":"C0376670",
"C0050252":"C0377562",
"C0339663":"C0549122",
"C2712054":"C0743330",
"C0557875":"C0849970",
"C0032305":"C1535939",
"C0337197":"C1579887",
"C0005001":"C1704272",
"C1960564":"C2020625",
"C0432747":"C2108211",
"C0151706":"C2979982",
"C0175677":"C3263723",
"C0232280":"C4049827",
"C1883552.":"C0004093",
"C00018946":"C0018946",
"C0040053.":"C0040053",
"CUi-less":"cui-less",
"Cui-less":"cui-less",
"CUI-less": "cui-less",
"C0003878":"C0003873",
"C002076":"C0020676",
"C00240066":"C0240066",
"C023529":"C0235329",
"C03203358":"C3203358"}

class FewShotPrompting:

    def __init__(
        self,
        spacy_model: str = "en_core_web_md",
        semeval_folder: str = None,
    ) -> None:
        """Initializes UMLS ontology based prompting model

        Args:
            spacy_model (str, optional): The spacy model to tokenize and split document into sentences. Defaults to en_core_web_md.
            vocabulary (str, optional): The vocabulary to use for generating prompts. Defaults to None.
            user (str, optional): The username for the UMLS database. Defaults to None.
            pwd (str, optional): The password for the UMLS database. Defaults to None.
            database (str, optional): The name of the UMLS database. Defaults to None.
            ip (str, optional): The IP address of the UMLS database. Defaults to None.
        """        
        super().__init__()

        self.vocabulary = vocabulary

        self.semeval_folder = semeval_folder

        self.nlp = spacy.load(spacy_model, disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])

        self.nlp.add_pipe("sentencizer")
                
    def generate(
        self,
        prompt: str,
        dataset: str,
        semantic_groups: List[str] = None,
        shots: int = 5,
        surrounding_sentences_count: int = 2,
    ) -> List[Dict[str, Any]]:
        """Generates prompts based on the provided template and example documents.

        Args:
            prompt (str): The template for generating prompts.
            dataset (str): A collection of example documents to use as demonstrations.
            semantic_groups (List[str], optional): A list of semantic groups to filter the entities. Defaults to None.
            shots (int, optional): The number of example documents to use for each prompt. Defaults to 5.
            surrounding_sentences (int, optional): The number of sentences surronding the mention for each example document. Defaults to 2.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the generated prompts.
        """

        if self.umls_folder is not None:
            entities = self._generate_entities_definition_local(semantic_groups)
        else:
            entities = self._generate_entities_definition_database(semantic_groups)
        
        ask_prompts = []

        for entity in tqdm(entities, desc="Cleaning up entities"):

            cleaned_syn = list(entities[entity]["synonym"]) 

            if len(cleaned_syn) == 0:
                entities[entity]["synonym"] = None
            else:
                entities[entity]["synonym"] = ",".join(cleaned_syn[:5])
            
            cleaned_def = list(entities[entity]["definition"])
            
            if len(cleaned_def) == 0:
                entities[entity]["definition"] = None
            else:
                entities[entity]["definition"] = cleaned_def[0]

            new_entity = {
                "id": entity,
                "entity": entities[entity]["entity"],
                "text": prompt.format(**entities[entity]),
            }

            ask_prompts.append(new_entity)
        
        finished_prompts = []

        mention_pool = self._generate_shots(dataset, surrounding_sentences_count, prompt, entities)

        for ask_prompt in tqdm(ask_prompts, desc="Generating prompts"):
            demonstrations = sample(mention_pool, shots)

            ask_prompt["text"] = "\n".join(demonstrations) + "\n" + ask_prompt["text"]
            finished_prompts.append(ask_prompt)
        
        return finished_prompts

    def _generate_shots(self, dataset, surrounding_sentences_count, prompt, entities):
        if dataset == "bc5cdr":
            return self._process_bc5cdr(surrounding_sentences_count, prompt, entities)
        elif dataset == "ncbi":
            return self._process_ncbi(surrounding_sentences_count, prompt, entities)
        else:
            return self._process_semeval(self.semeval_folder, surrounding_sentences_count, prompt, entities)
    
    def _process_bc5cdr(self):

        dataset = load_dataset("bigbio/bc5cdr")

        passages = dataset["train"]["passages"]

        for passage in tqdm(passages):
            text = passage[0]["text"]+ " " + passage[1]["text"]

            entities = passage[0]["entities"]

            doc = self.nlp(text)

            for entity in entities:
                if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                    offsets = entity["offsets"]

                    normalized = entity["normalized"]

                    for offset, normalized_item in zip(offsets, normalized):

                        mapped_cui = self._map_to_cui(normalized_item["db_id"])

                        synoynms = self._get_syn(mapped_cui)

                        syn_replaced_texts = self._syn_replace(doc, offset[0], offset[1], synoynms)

                        for syn_replaced_text in syn_replaced_texts:
                            csv_file.writerow([mapped_cui, syn_replaced_text])
            
    def _process_ncbi(self, surrounding_sentences_count, prompt, entities):

        dataset = load_dataset("bigbio/ncbi_disease")

        mentions = dataset["train"]["mentions"]

        titles = dataset["train"]["title"]

        abstracts = dataset["train"]["abstract"]

        shots = []

        for title, abstract, list_of_mentions in tqdm(zip(titles, abstracts, mentions)):
            text = title + " " + abstract

            mentions = []

            for mention in list_of_mentions:
                if "+" in mention["concept_id"]:
                    concept_ids_split = mention["concept_id"].split("+")
                else:
                    concept_ids_split = mention["concept_id"].split("|")

                offset = mention["offsets"]

                for concept_id_split in concept_ids_split:
                    if "OMIM" in concept_id_split:
                        concept_id_split = concept_id_split.replace("OMIM:", "")
                    
                    concept_id_split = concept_id_split.strip()

                    mentions.append((offset[0], offset[1], concept_id_split))
            
            shots.extend(self._generate_mentions(text, mentions, surrounding_sentences_count, prompt, entities))

        return shots

    
    def _split_offsets(self, string_input: str) -> np.ndarray:
        return np.array([x.split("-") for x in string_input.split(",")]).astype(int)
                        
    def _process_semeval(self, input_folder: str):
        list_of_pipe_files = [os.path.join(input_folder, x)
                                for x in os.listdir(input_folder) if x.find('pipe') >= 0]

        for filename in tqdm(list_of_pipe_files):
            df = pd.read_csv(filename, sep='|', header=None, on_bad_lines='warn')

            df[2] = df[2].fillna('cui-less')

            df[2] = df[2].replace(CUI_FIX)

            df = df[df[2] != "cui-less"]

            if len(df) != 0:

                with open(os.path.join(input_folder, df[0].iloc[0])) as f:
                    text_document = f.read()

                cuis = list(df[2])
                offsets = list(df[1])

                doc = self.nlp(text_document)

                for cui, offset_group in zip(cuis, offsets):
                    split_offset_group = self._split_offsets(offset_group)

                    for offset in split_offset_group:
                        synoynms = self._get_syn(cui)

                        syn_replaced_texts = self._syn_replace(doc, offset[0], offset[1], synoynms)

    def _generate_mentions(self, document: str, mentions: List[Tuple], surrounding_sentences_count: int, prompt: str, entities: Dict[str, Dict]):
        output_prompt = prompt + " {output}"

        mention_pool = []

        doc = self.nlp(document)

        sentences = [(sent.start, sent.end, i) for i, sent in enumerate(doc.sents)]

        sentence_tree = IntervalTree.from_tuples(sentences)

        for mention in mentions:

            start, end, id = mention

            entity = document[start:end]

            if self.vocabulary is None and id in self.old_to_new_cui:
                id = self.old_to_new_cui[id]
            
            elif self.vocabulary is not None:
                id = self._map_code(id)
            
            try:

                definition = entities[id]["definition"]
                synonyms = entities[id]["synonym"]
            
            except KeyError as e:
                print(e)
                print(f"Error with {id}")
                continue

            span = doc.char_span(start,end, alignment_mode='expand')

            token_index_start, token_index_end = span.start, span.end
            
            if token_index_start == token_index_end:
                sentences_containing_mention = [tuple(x) for x in list(sentence_tree[token_index_start])]
            else:
                sentences_containing_mention = [tuple(x) for x in list(sentence_tree[token_index_start: token_index_end])]
            
            left_bound_idx = max(min(sentences_containing_mention)[2]-surrounding_sentences_count,0)
            right_bound_idx = min(max(sentences_containing_mention)[2]+surrounding_sentences_count,len(sentence_tree))

            sentence_idxs = sentences[left_bound_idx:right_bound_idx]

            left_bound = min(sentence_idxs)[0]
            right_bound = max(sentence_idxs)[1]

            surrounding_sentences = [tuple(x) for x in list(sentence_tree[left_bound:right_bound])]

            left_surronding_bound_token = min(surrounding_sentences)[0]
            left_surronding_bound_character = doc[left_surronding_bound_token].idx

            right_surronding_bound_token = max(surrounding_sentences)[1]

            if right_surronding_bound_token == len(doc):
                right_surronding_bound_token -= 1

            text = doc[left_surronding_bound_token:right_surronding_bound_token].text

            start = start-left_surronding_bound_character

            end = end-left_surronding_bound_character

            clean_text = copy.copy(text[:start]) + f' <1CUI> ' + copy.copy(text[start:end]) + f' <1CUI> ' + copy.copy(text[end:])

            if synonyms is None:
                synonyms = "No Synonyms"

            if definition is None:
                definition = "No Definition"

            input_example = {
                "id": id,
                "entity": entity,
                "synonym": synonyms,
                "definition": definition,
                "output": clean_text,
            }

            mention_pool.append(output_prompt.format(**input_example))
        
        return mention_pool