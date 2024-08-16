import os

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def ood_analysis(dataset_folder: str, results_folder: str):
    conll_files = [os.path.join(results_folder, x) for x in os.listdir(results_folder)]

    semeval_train = pd.read_csv(os.path.join(dataset_folder, "semeval/train_semeval_cui.conll"), sep="\s", header = None, names = ["token", "label", "cui"])

    train_cuis = set(semeval_train.cui[(semeval_train.cui != "NOCUI") & (semeval_train.cui != "CUI-less")].unique())
    train_cuis.add("CUI-less")

    semeval_test_gold = pd.read_csv(os.path.join(dataset_folder, "semeval/test_semeval_cui.conll"), sep="\s", header = None, names = ["token", "label", "cui"])

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


# def per_cui_ood_analysis(dataset_folder: str, results_folder: str, generated_output: str):

#     conll_files = [os.path.join(results_folder, x) for x in os.listdir(results_folder)]

#     semeval_train = pd.read_csv(os.path.join(dataset_folder, "semeval_cui/trainsemeval.conll"), sep="\s", header = None, names = ["token", "label", "cui"])

#     train_cuis = set(semeval_train.cui[(semeval_train.cui != "NOCUI") & (semeval_train.cui != "CUI-less")].unique())
#     train_cuis.add("CUI-less")

#     semeval_test = pd.read_csv(os.path.join(dataset_folder, "semeval_cui/testsemeval.conll"), sep="\s", header = None, names = ["token", "label", "cui"])

#     test_cuis = set(semeval_test.cui[(semeval_test.cui != "NOCUI") & (semeval_test.cui != "CUI-less")].unique())

#     cuis_generated = pd.read_csv(generated_output, header = None, names= ["cui", "default", "text"])

#     generated_cuis = set(cuis_generated.cui[(cuis_generated.cui != "NOCUI") & (cuis_generated.cui != "CUI-less")].unique())

#     ood_cuis = test_cuis - train_cuis

#     overlap = generated_cuis & ood_cuis

#     ood_not_gen = ood_cuis - generated_cuis

#     total_correct = pd.Series([0] * len(overlap), index = list(overlap))

#     for conll_file in conll_files[2:]:
#         df = pd.read_csv(conll_file, sep='\s', engine='python', header=None, names = ['token', 'label','predicted', 'confidence', 'r'])

#         df['cui'] = semeval_test['cui']

#         df_ood = df[df['cui'].isin(overlap)].reset_index(drop = True, inplace = False)

#         df_ood['correct'] = np.where(df_ood['predicted'] == df_ood['label'], 1, 0)

#         ood_correct = df_ood.groupby('cui')[['cui', 'correct']].agg(sum)

#         ood_correct = np.where(ood_correct['correct'] > 0, 1, 0)

#         total_correct = total_correct + ood_correct

#         print(conll_file, ood_correct.sum()/len(ood_correct))