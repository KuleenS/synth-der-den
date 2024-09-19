import csv

import os

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def ood_analysis(dataset: str, dataset_folder: str, results_folder: str):
    conll_files = [os.path.join(results_folder, x) for x in os.listdir(results_folder) if dataset in x]

    semeval_train = pd.read_csv(os.path.join(dataset_folder, f"{dataset}/train_{dataset}_cui.conll"), delim_whitespace=True, header = None, names = ["token", "label", "cui"], quoting=csv.QUOTE_NONE)

    train_cuis = set(semeval_train.cui[(semeval_train.cui != "NOCUI") & (semeval_train.cui != "CUI-less")].unique())
    train_cuis.add("CUI-less")

    for conll_file in conll_files:
        print(os.path.basename(conll_file), "results")
        
        df = pd.read_csv(conll_file, sep='\s', engine='python', header=None, names = ['token', 'label','predicted', 'cui'], on_bad_lines='skip')

        df_ood = df[~df['cui'].isin(train_cuis)]

        pu,ru,fu,su = precision_recall_fscore_support(df_ood['label'], df_ood['predicted'], labels = ["DISEASE", "O"])

        pm,rm,fm,sm = precision_recall_fscore_support(df_ood['label'], df_ood['predicted'], labels = ["DISEASE", "O"])

        acc = accuracy_score(df_ood['label'], df_ood['predicted'])

        correct = accuracy_score(df_ood['label'], df_ood['predicted'], normalize = False)

        print("OOD Analysis")
        print("Micro Prec: {} | Micro Rec: {} | Micro F1: {} | Support: {}".format(pu, ru, fu, su))
        print("Macro Prec: {} | Macro Rec: {} | Macro F1: {}".format(pm, rm, fm))
        print("Accuracy: {}".format(acc))
        print("Correct: {}".format(correct))
        print("Total: {}".format(len(df_ood)))

        print("Overall Analysis")
        pu,ru,fu,su = precision_recall_fscore_support(df['label'], df['predicted'], labels = ["DISEASE", "O"])

        pm,rm,fm,sm = precision_recall_fscore_support(df['label'], df['predicted'], labels = ["DISEASE", "O"])

        acc = accuracy_score(df['label'], df['predicted'])
        
        correct = accuracy_score(df['label'], df['predicted'], normalize = False)

        print("Micro Prec: {} | Micro Rec: {} | Micro F1: {} | Support: {}".format(pu, ru, fu, su))
        print("Macro Prec: {} | Macro Rec: {} | Macro F1: {}".format(pm, rm, fm))
        print("Accuracy: {}".format(acc))
        print("Correct: {}".format(correct))
        print("Total: {}".format(len(df_ood)))
