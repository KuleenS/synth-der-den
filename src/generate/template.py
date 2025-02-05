import csv

import os

import mysql.connector
import pandas as pd 

from tqdm import tqdm

class Templating:

    def __init__(self, output_file: str) -> None:

        self.output_file = output_file
        self.user = os.environ['UMLS_USER']
        self.pwd = os.environ['UMLS_PWD']
        self.database = os.environ['UMLS_DATABASE_NAME']
        self.ip = os.environ['UMLS_IP']

        self.template = """Patient Name: [Patient Name 2301]
Chief Complaint: Symptoms related to <1CUI>{disease}</1CUI>
History of Present Illness: Patient presents with signs and symptoms consistent with <1CUI>{disease}</1CUI>. Further evaluation and diagnostic workup are in progress to confirm severity and appropriate management.
Plan: Proceed with necessary assessments and initiate appropriate care as indicated."""

    def generate(self):

        cnx = mysql.connector.connect(user=self.user, password=self.pwd, database=self.database, host=self.ip)

        query = ("SELECT MRCONSO.CUI, MRCONSO.STR, MRDEF.DEF FROM MRCONSO \
        JOIN MRSTY ON (MRCONSO.CUI = MRSTY.CUI) \
        JOIN MRDEF ON (MRCONSO.CUI = MRDEF.CUI) \
        WHERE LAT='ENG' AND ISPREF='Y' \
        AND TUI IN ('T047','T020','T190','T049','T019','T050','T033','T037','T048','T191','T046','T184') \
        GROUP BY MRCONSO.CUI ORDER BY MRCONSO.CUI;") 

        df = pd.read_sql(query, con=cnx)

        with open(self.output_file, "w") as f:
            csv_file = csv.writer(f)

            csv_file.writerow(["cui", "matched_output"])

            for index, row in tqdm(df.iterrows()):
                csv_file.writerow([row["CUI"], self.template.format(disease=row["STR"])])