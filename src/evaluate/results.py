import csv

import os

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def ood_analysis(dataset: str, dataset_folder: str, results_folder: str, run_number: int, mode: int):
    conll_files = [os.path.join(results_folder, x) for x in os.listdir(results_folder) if dataset in x and ".conll" in x]

    semeval_train = pd.read_csv(os.path.join(dataset_folder, f"{dataset}/train_{dataset}_cui.conll"), delim_whitespace=True, header = None, names = ["token", "label", "cui"], quoting=csv.QUOTE_NONE)

    train_cuis = set(semeval_train.cui[(semeval_train.cui != "NOCUI") & (semeval_train.cui != "CUI-less")].unique())
    train_cuis.add("CUI-less")

    with open(os.path.join(results_folder, f"{dataset}_mode_{mode}_run_number_{run_number}_ood.csv"), "w") as f:

        csv_out = csv.writer(f)

        csv_out.writerow(['filename', 'precision DISEASE', 'precision O', 'recall DISEASE', 'recall O', 'f1 DISEASE', 'f1 O', 'support DISEASE', 'support O', 'accuracy', 'correct', 'length'])

        for conll_file in conll_files:
            df = pd.read_csv(conll_file, sep='\s', engine='python', header=None, names = ['token', 'label','predicted', 'cui'], on_bad_lines='skip')

            df_ood = df[~df['cui'].isin(train_cuis)]

            pu, ru, fu, su = precision_recall_fscore_support(df_ood['label'], df_ood['predicted'], labels = ["DISEASE", "O"])

            pu_d, pu_o, ru_d, ru_o, fu_d, fu_o, su_d, su_o = list(pu) +  list(ru) + list(fu) + list(su)

            # pm,rm,fm,sm = precision_recall_fscore_support(df_ood['label'], df_ood['predicted'], labels = ["DISEASE", "O"])

            acc = accuracy_score(df_ood['label'], df_ood['predicted'])

            correct = accuracy_score(df_ood['label'], df_ood['predicted'], normalize = False)

            csv_out.writerow([os.path.basename(conll_file), pu_d,pu_o,ru_d,ru_o,fu_d,fu_o, su_d, su_o, acc, correct, len(df_ood)])

        # print("Overall Analysis")
        # pu,ru,fu,su = precision_recall_fscore_support(df['label'], df['predicted'], labels = ["DISEASE", "O"])

        # pm,rm,fm,sm = precision_recall_fscore_support(df['label'], df['predicted'], labels = ["DISEASE", "O"])

        # acc = accuracy_score(df['label'], df['predicted'])
        
        # correct = accuracy_score(df['label'], df['predicted'], normalize = False)

        # print("Micro Prec: {} | Micro Rec: {} | Micro F1: {} | Support: {}".format(pu, ru, fu, su))
        # print("Macro Prec: {} | Macro Rec: {} | Macro F1: {}".format(pm, rm, fm))
        # print("Accuracy: {}".format(acc))
        # print("Correct: {}".format(correct))
        # print("Total: {}".format(len(df_ood)))
