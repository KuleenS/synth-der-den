from pathlib import Path 

import csv

import os

import mysql.connector
import polars as pl


from tqdm import tqdm

class Templating:

    def __init__(self, output_file: str, umls_folder: str = None) -> None:

        self.output_file = output_file
        self.umls_folder = umls_folder
        self.user = os.environ['UMLS_USER']
        self.pwd = os.environ['UMLS_PWD']
        self.database = os.environ['UMLS_DATABASE_NAME']
        self.ip = os.environ['UMLS_IP']

        self.template = """Patient Name: [Patient Name 2301]
Chief Complaint: Symptoms related to <1CUI>{disease}</1CUI>
History of Present Illness: Patient presents with signs and symptoms consistent with <1CUI>{disease}</1CUI>. Further evaluation and diagnostic workup are in progress to confirm severity and appropriate management.
Plan: Proceed with necessary assessments and initiate appropriate care as indicated."""

    def generate(self):

        if self.umls_folder is None:

            cnx = mysql.connector.connect(user=self.user, password=self.pwd, database=self.database, host=self.ip)

            query = ("SELECT MRCONSO.CUI, MRCONSO.STR, MRDEF.DEF FROM MRCONSO \
            JOIN MRSTY ON (MRCONSO.CUI = MRSTY.CUI) \
            JOIN MRDEF ON (MRCONSO.CUI = MRDEF.CUI) \
            WHERE LAT='ENG' AND ISPREF='Y' \
            AND TUI IN ('T047','T020','T190','T049','T019','T050','T033','T037','T048','T191','T046','T184') \
            GROUP BY MRCONSO.CUI ORDER BY MRCONSO.CUI;") 

            df = pl.read_database(query, connection=cnx)
        
        else:
            root = Path(self.umls_folder)

            mrconso = root / "MRCONSO.RRF"

            mrsty = root / "MRSTY.RRF"

            MRCONSO = pl.read_csv(mrconso, separator="|", has_header=False, encoding="utf8", quote_char=None)

            MRSTY = pl.read_csv(mrsty, separator="|", has_header=False, encoding="utf8", quote_char=None)

            MRCONSO.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF", "BLANK"]

            MRSTY.columns = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "BLANK"]

            MRSTY_FILTER = MRSTY.filter(pl.col('TUI').str.contains_any(['T047','T020','T190','T049','T019','T050','T033','T037','T048','T191','T046','T184']))[["CUI", "TUI"]]

            MRCONSO_FILTER = MRCONSO.filter((pl.col("LAT") == "ENG") & (pl.col("ISPREF") == "Y"))[["CUI", "STR"]]

            MRCONSO_JOIN_MRSTY = MRCONSO_FILTER.join(MRSTY_FILTER, left_on="CUI", right_on="CUI", how="inner")

            df = MRCONSO_JOIN_MRSTY.unique("CUI")

            print(df)

        with open(self.output_file, "w") as f:
            csv_file = csv.writer(f)

            csv_file.writerow(["cui", "matched_output"])

            for row in tqdm(df.iter_rows(named=True)):
                
                csv_file.writerow([row["CUI"], self.template.format(disease=row["STR"])])