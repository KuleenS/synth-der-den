import csv

import os

import mysql.connector
import pandas as pd 

from tqdm import tqdm

from utils import get_umls_data

class InputData:
    def __init__(self, output_file: str) -> None:

        self.output_file = output_file
        self.user = os.environ['UMLS_USER']
        self.pwd = os.environ['UMLS_PWD']
        self.database = os.environ['UMLS_DATABASE_NAME']
        self.ip = os.environ['UMLS_IP']

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

            csv_file.writerow(["CUI", "STR", "PROMPT"])

            for index, row in tqdm(df.iterrows()):

                umls_data = get_umls_data(row['CUI'], cnx)

                for key in umls_data:
                    umls_data[key] = [x.replace('"', "") for x in umls_data[key]]
                
                prepared_cui = row['CUI'].replace('"', "")
                prepared_str = row['STR'].replace('"', "")
                prepared_def = row['DEF'].replace('"', "")

                umls_names = ' '.join(list(set(umls_data['UMLS_NAME'])))
                
                input_text = f"Pretend you are a physician: Write a clinical note for a patient that mentions the condition {prepared_str} either explicity or as a synoynm or abbreviation to this condition. It is also known as {umls_names}. It is defined as {prepared_def}. Place tokens <1CUI> before and after the mention of this condition. For example <1CUI> {prepared_str} <1CUI>."

                csv_file.writerow([prepared_cui, prepared_str, input_text])