import csv

from pathlib import Path

import os

import mysql.connector
import pandas as pd 

import polars as pl

from tqdm import tqdm

from src.utils import get_umls_data

class InputData:
    def __init__(self, output_file: str, umls_folder: str = None) -> None:

        self.output_file = output_file
        self.user = os.environ['UMLS_USER']
        self.pwd = os.environ['UMLS_PWD']
        self.database = os.environ['UMLS_DATABASE_NAME']
        self.ip = os.environ['UMLS_IP']

        self.umls_folder = umls_folder

        self.cnx = mysql.connector.connect(user=self.user, password=self.pwd, database=self.database, host=self.ip)
    
    def _generate_sql(self):
        query = ("SELECT MRCONSO.CUI, MRCONSO.STR, MRDEF.DEF FROM MRCONSO \
        JOIN MRSTY ON (MRCONSO.CUI = MRSTY.CUI) \
        JOIN MRDEF ON (MRCONSO.CUI = MRDEF.CUI) \
        WHERE LAT='ENG' AND ISPREF='Y' \
        AND TUI IN ('T047','T020','T190','T049','T019','T050','T033','T037','T048','T191','T046','T184') \
        GROUP BY MRCONSO.CUI ORDER BY MRCONSO.CUI;") 

        df = pd.read_sql(query, con=self.cnx)

        with open(self.output_file, "w") as f:
            csv_file = csv.writer(f)

            csv_file.writerow(["CUI", "STR", "PROMPT"])

            for _, row in tqdm(df.iterrows()):

                umls_data = get_umls_data(row['CUI'], self.cnx)

                for key in umls_data:
                    umls_data[key] = [x.replace('"', "") for x in umls_data[key]]
                
                prepared_cui = row['CUI'].replace('"', "")
                prepared_str = row['STR'].replace('"', "")
                prepared_def = row['DEF'].replace('"', "")

                umls_names = ' '.join(list(set(umls_data['UMLS_NAME'])))
                
                input_text = f"Pretend you are a physician: Write a clinical note for a patient that mentions the condition {prepared_str} either explicity or as a synoynm or abbreviation to this condition. It is also known as {umls_names}. It is defined as {prepared_def}. Place tokens <1CUI> before and after the mention of this condition. For example <1CUI> {prepared_str} <1CUI>."

                csv_file.writerow([prepared_cui, prepared_str, input_text])
    
    def _generate_local(self):

        root = Path(self.umls_folder)

        mrconso = root / "MRCONSO.RRF"

        mrsty = root / "MRSTY.RRF"

        mrdef = root / "MRDEF.RRF"

        MRCONSO = pl.read_csv(mrconso, separator="|", has_header=False, encoding="utf8", quote_char=None)

        MRSTY = pl.read_csv(mrsty, separator="|", has_header=False, encoding="utf8", quote_char=None)

        MRDEF = pl.read_csv(mrdef, separator="|", has_header=False, encoding="utf8", quote_char=None)

        MRCONSO.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF", "BLANK"]

        MRSTY.columns = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "BLANK"]

        MRDEF.columns = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF", "BLANK"]

        MRCONSO = MRCONSO.filter((pl.col("LAT") == "ENG") & (pl.col("ISPREF") == "Y"))[["CUI", "STR"]]

        semantic_groups = ['T047','T020','T190','T049','T019','T050','T033','T037','T048','T191','T046','T184']

        MRSTY = MRSTY.filter(pl.col('TUI').str.contains_any(semantic_groups))[["CUI", "TUI"]]

        MRDEF = MRDEF[["CUI", "DEF"]]

        JOINED = MRCONSO.join(MRSTY, left_on="CUI", right_on="CUI", how="inner").join(MRDEF, left_on="CUI", right_on="CUI", how="inner")

        output = JOINED[["CUI", "STR", "DEF"]]

        output = output.unique(subset=["CUI"])

        with open(self.output_file, "w") as f:
            csv_file = csv.writer(f)

            csv_file.writerow(["CUI", "STR", "PROMPT"])

            for row in tqdm(output.iter_rows(named=True)):

                umls_data = get_umls_data(row['CUI'], self.cnx)

                for key in umls_data:
                    umls_data[key] = [x.replace('"', "") for x in umls_data[key]]
                
                prepared_cui = row['CUI'].replace('"', "")
                prepared_str = row['STR'].replace('"', "")
                prepared_def = row['DEF'].replace('"', "")

                umls_names = ' '.join(list(set(umls_data['UMLS_NAME'])))
                
                input_text = f"Pretend you are a physician: Write a clinical note for a patient that mentions the condition {prepared_str} either explicity or as a synoynm or abbreviation to this condition. It is also known as {umls_names}. It is defined as {prepared_def}. Place tokens <1CUI> before and after the mention of this condition. For example <1CUI> {prepared_str} <1CUI>."

                csv_file.writerow([prepared_cui, prepared_str, input_text])

    def generate(self):
        if self.umls_folder is not None: 
            self._generate_local()
        else:
            self._generate_sql()

       