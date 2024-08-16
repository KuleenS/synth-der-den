from typing import List, Tuple

from datasets import load_dataset

class NCBI:
    def __init__(self):
        self.dataset = load_dataset("bigbio/ncbi_disease")
    
    def generate_data(self, split: str ="train") -> List[Tuple[str, str, str]]:
        mentions = self.dataset[split]["mentions"]

        results = []

        for mention in mentions:

            for concept_id in mention:

                concept_ids_split = concept_id["concept_id"].split("|")

                text = concept_id["text"]

                for concept_id_split in concept_ids_split:
                    if "OMIM" in concept_id_split:
                        concept_id_split = concept_id_split.replace("OMIM:", "")

                        results.append([text, concept_id_split, "OMIM"])
                    else:
                        results.append([text, concept_id_split, "MESH"])
        
        return results
