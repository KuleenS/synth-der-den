import os
from typing import List, Tuple, Iterable, Dict
import pandas as pd
import medspacy
from spacy.tokens import Doc
import numpy as np
import csv

import mysql
from tqdm import tqdm
import mysql.connector
from intervaltree import IntervalTree

def get_umls_data(cui: str, connector: mysql.connector) -> Dict[str, Tuple[str]]:
    data = {'UMLS_NAME': None, 'MODE': None, 'MAX': None, 'MIN': None, 'DEF': None}
    cursor = connector.cursor(buffered=True)
    query = (f"SELECT STR FROM MRCONSO WHERE CUI='{cui}' AND LAT='ENG' AND ISPREF = 'Y' GROUP BY STR")
    cursor.execute(query)

    # gets the UMLS name if it has it
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['UMLS_NAME'] = temp
        else:
            data['UMLS_NAME'] = ('',)
    except:
        data['UMLS_NAME'] = ('',)

    # returns the mode of the names if it has one
    query = (f"SELECT STR \
            FROM MRCONSO WHERE CUI='{cui}' AND LAT='ENG' \
            GROUP BY STR ORDER BY COUNT(LENGTH(STR)) DESC;")
    cursor.execute(query)
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['MODE'] = temp
        else:
            data['MODE'] = ('',)
    except:
        data['MODE'] = ('',)

    # returns the longest length name in UMLS if it has one
    query = (f"SELECT STR FROM MRCONSO \
        WHERE CUI='{cui}' AND LAT='ENG' \
        AND LENGTH(STR) = \
        (SELECT MAX(LENGTH(STR)) FROM MRCONSO \
        WHERE CUI='{cui}' AND LAT='ENG')")
    cursor.execute(query)
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['MAX'] = temp
        else:
            data['MAX'] = ('',)
    except:
        data['MAX'] = ('',)

    # returns the shortest length name in UMLS if it has one
    query = (f"SELECT STR FROM MRCONSO \
        WHERE CUI='{cui}' AND LAT='ENG' \
        AND LENGTH(STR) = \
        (SELECT MIN(LENGTH(STR)) FROM MRCONSO \
        WHERE CUI='{cui}' AND LAT='ENG')")
    cursor.execute(query)
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['MIN'] = temp
        else:
            data['MIN'] = ('',)
    except:
        data['MIN'] = ('',)
    
    # returns the definition in UMLS if it has one
    query = (f"SELECT DEF FROM MRDEF WHERE CUI='{cui}' LIMIT 1")
    cursor.execute(query)
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['DEF'] = temp
        else:
            data['DEF'] = ('',)
    except:
        data['DEF'] = ('',)
    cursor.close()
    return data


class GenerateSemevalData:
    def __init__(self, input_folder: str, output_folder: str) -> None:

        self.input_folder = input_folder
        self.output_folder = output_folder

        self.user = os.environ['UMLS_USER']
        self.pwd = os.environ['UMLS_PWD']
        self.database = os.environ['UMLS_DATABASE_NAME']
        self.ip = os.environ['UMLS_IP']

    def split_offsets(self, string_input: str) -> np.ndarray:
        return np.array([x.split('-') for x in string_input.split(',')]).astype(int)

    def create_input_text(self, mention: str, cui: str, connector) -> str:
        # if its cuiless do not do anything special
        if cui == 'CUI-less':
            input_text = f"Pretend you are a physician: Write a clinical note for a patient that mentions the condition {mention} either explicity or as a synoynm or abbreviation to this condition. Place tokens <1CUI> before and after the mention of this condition. For example <1CUI> {mention} <1CUI>."
            return input_text

        umls_data = get_umls_data(cui, connector)

        umls_names = ' '.join(list(set(umls_data['UMLS_NAME'])))
        input_text = f"Pretend you are a physician: Write a clinical note for a patient that mentions the condition {mention} either explicity or as a synoynm or abbreviation to this condition. It is also known as {umls_names}. It is defined as {umls_data['DEF'][0]}. Place tokens <1CUI> before and after the mention of this condition. For example <1CUI> {mention} <1CUI>."

        return input_text

    def create_sentence_offsets(self, doc: Doc) -> List[Tuple[int, int]]:
        # ['This is a test sentence for show.','This is a second test sentence.','This is a third test sentence.']
        sentences: List[str] = [i.text for i in doc.sents]
        # creates a cumulative sum of the offsets of each sentence
        cumulative_sum: Iterable[int] = np.cumsum([len(x + " ") for x in sentences])  # array([34, 66, 97])

        # creates a tuples off the offsets of the sentence
        tuple_list = []
        for i in range(len(cumulative_sum)):
            if i - 1 == -1:
                tuple_range: Tuple[int, int] = (0, cumulative_sum[i])
            else:
                tuple_range: Tuple[int, int] = (cumulative_sum[i - 1], cumulative_sum[i])
            tuple_list.append(tuple_range)

        return tuple_list

    def process_files(self, input_files: List[str], split: str, nlp, cnx: mysql.connector.MySQLConnection) -> None:
        for filename in tqdm(input_files):

            clean_input = []
            clean_output = []

            df = pd.read_csv(filename, sep='|', header=None, on_bad_lines='warn')
            df[2] = df[2].fillna('CUI-less')

            with open(os.path.join(self.input_folder, df[0][0])) as f:
                text_document = f.read()

            text_document = text_document.replace("\n", " ").replace("\t", " ")

            doc = nlp(text_document)

            raw_offsets: List[List[int]] = df[1].apply(self.split_offsets).tolist()
            offsets: List[int] = [item for sublist in raw_offsets for item in sublist]

            raw_cuis: List[str] = df[2].tolist()
            multiplied_cuis: List[List[str]] = [len(offset)*[CUI] for offset, CUI in zip(offsets, raw_cuis)]
            CUIS: List[str] = [item for sublist in multiplied_cuis for item in sublist]

            token_tree = IntervalTree.from_tuples([(x.idx, x.idx + len(x.text_with_ws), x) for x in doc])

            sentences = [(sent.start, sent.end, i) for i, sent in enumerate(doc.sents)]

            sentence_tree = IntervalTree.from_tuples(sentences)

            for offset, CUI in zip(offsets, CUIS):
                tokens = sorted(list(token_tree[offset[0]:offset[1]]), key=lambda x: x[0])

                token_index_start = tokens[0][2].i
                token_index_end = tokens[-1][2].i

                mention = doc.text[offset[0]:offset[1]]

                prompt = self.create_input_text(mention, CUI, cnx)

                clean_input.append(prompt)

                if token_index_start == token_index_end:
                    sentences_containing_mention = [tuple(x) for x in list(sentence_tree[token_index_start])]
                else:
                    sentences_containing_mention = [tuple(x) for x in list(sentence_tree[token_index_start: token_index_end])]

                left_bound_idx = max(min(sentences_containing_mention)[2]-2,0)
                right_bound_idx = min(max(sentences_containing_mention)[2]+2,len(sentence_tree))

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

                start = offset[0]-left_surronding_bound_character
                end = offset[1]-left_surronding_bound_character
                text = text[:start] + f' <1CUI> ' + text[start:end] + f' <1CUI> ' + text[end:]

                clean_output.append(text)

            with open(os.path.join(self.output_folder, split, f'{os.path.basename(filename)[:-5]}.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(clean_input, clean_output))

    def generate(self) -> None:
        list_of_pipe_files = [os.path.join(self.input_folder, x)
                              for x in os.listdir(self.input_folder) if x.find('pipe') >= 0]

        split_1 = int(0.8 * len(list_of_pipe_files))
        split_2 = int(0.9 * len(list_of_pipe_files))
        train_filenames = list_of_pipe_files[:split_1]
        dev_filenames = list_of_pipe_files[split_1:split_2]
        test_filenames = list_of_pipe_files[split_2:]

        nlp = medspacy.load()

        cnx: mysql.connector.MySQLConnection = mysql.connector.connect(user=self.user, password=self.pwd, database=self.database, host=self.ip)

        self.process_files(train_filenames, 'traindata', nlp, cnx)

        self.process_files(dev_filenames, 'devdata', nlp, cnx)

        self.process_files(test_filenames, 'testdata', nlp, cnx)
