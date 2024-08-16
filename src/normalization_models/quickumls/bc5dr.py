from typing import List, Tuple

from datasets import load_dataset

class BC5CDR:
    def __init__(self):
        self.dataset = load_dataset("bigbio/bc5cdr")
    
    def generate_data(self, split: str ="train") -> List[Tuple[str, str, str]]:
        passages = self.dataset[split]["passages"]

        results = []

        for passage in passages:

            entities = passage[0]["entities"]

            for entity in entities:
                if entity["type"] == "Disease" and len(entity["normalized"]) != 0:
                    results.append([entity["text"][0], entity["normalized"][0]["db_id"], entity["normalized"][0]["db_name"]])
            
            entities = passage[1]["entities"]

            for entity in entities:
                if entity["type"] == "Disease" and len(entity["normalized"]) != 0:
                    results.append([entity["text"][0], entity["normalized"][0]["db_id"], entity["normalized"][0]["db_name"]])
        
        return results
