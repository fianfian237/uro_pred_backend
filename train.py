import pandas as pd
import numpy as np

import graphviz

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from preprocessing import ItemSelector
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, precision_score,
                             classification_report, f1_score, precision_recall_fscore_support, accuracy_score)

from scipy.stats import reciprocal, uniform
from joblib import dump

dataset_grade = pd.read_excel("../data/GRADE.xlsx")
dataset_stade = pd.read_excel("../data/STADE.xlsx")

columns_grade = ["Repondant", "Numéro", "Video", "Structure", "Size", "Number", "Lesion Margin", "Lesion pedicle",
           "Lesion fronds", "Vascular architecture of the bladder wall", "Microvascular architecture of the tumor",
           "Réponse GRADE", "Correction", "GRADE", "Note grade"]

columns_stade = ["Repondant", "Numéro", "Video", "Structure", "Size", "Number", "Lesion Margin", "Lesion pedicle",
           "Lesion fronds", "Vascular architecture of the bladder wall", "Microvascular architecture of the tumor",
           "Réponse STADE", "Correction", "STADE", "Note stade"]

dataset_grade.columns = columns_grade
dataset_stade.columns = columns_stade

def replace_nan_stade(video, col, col_value):
    """
    There are lot of nan like values in the training data. We write this function in order to replace each nan by the
    most frequent value experts gave to the feature in the corresponding video
    :param video: the video of the observation
    :param col: the name of the column
    :param col_value: the value we want to replace
    :return: given a video number and the value we want to replace, return the most frequent value experts gave to a
     feature in the video
    """
    if str(col_value).lower().strip() in ["cannot access", "not applicable"]:
        return dataset_stade[dataset_stade["Video"]==str(video)][col].mode()[0]
    else:
        return str(col_value)

def replace_nan_grade(video, col, col_value):
    """
    There are lot of nan like values in the training data. We write this function in order to replace each nan by the
    most frequent value experts gave to the feature in the corresponding video
    :param video: the video of the observation
    :param col: the name of the column
    :param col_value: the value we want to replace
    :return: given a video number and the value we want to replace, return the most frquent value experts gave to a
     feature in the video
    """
    if str(col_value).lower().strip() in ["cannot access", "not applicable"]:
        return dataset_grade[dataset_grade["Video"]==str(video)][col].mode()[0]
    else:
        return str(col_value)


for col in ["Structure", "Size", "Number", "Lesion Margin", "Lesion pedicle",
           "Lesion fronds", "Vascular architecture of the bladder wall", "Microvascular architecture of the tumor"]:
    dataset_grade[col] = dataset_grade.apply(lambda x: replace_nan_grade(x["Video"], col, x[col]), axis=1)
    dataset_stade[col] = dataset_stade.apply(lambda x: replace_nan_stade(x["Video"], col, x[col]), axis=1)


def clean_lesion_pedicle(value):
    if str(value) in ['Thin (< 1/3 tumor diameter)', 'Thin  (< 1/3 tumor diameter)']:
        return "Thin (< 1/3 tumor diameter)"
    else:
        return str(value)

dataset_grade["Lesion pedicle"] = dataset_grade.apply(lambda x: clean_lesion_pedicle(x["Lesion pedicle"]), axis=1)
dataset_stade["Lesion pedicle"] = dataset_stade.apply(lambda x: clean_lesion_pedicle(x["Lesion pedicle"]), axis=1)


def get_true_stade(col):
    inter = str(col).split()[0]
    if inter in ["T1", "T2"]:
        return "T1+T2"
    else:
        return inter

def get_true_grade(col):
    return str(col).split()[1]

def convert_rep_grade(col):
    return "BG" if str(col)== "Low grade" else "HG"

def convert_rep_stade(col):
    return "Ta" if str(col)=="Ta" else "T1+T2"

dataset_stade["true_stade"] = dataset_stade.apply(lambda x: get_true_stade(x["Correction"]), axis=1)
dataset_stade["reponse_stade_converted"] = dataset_stade.apply(lambda x: convert_rep_stade(x["Réponse STADE"]), axis=1)

dataset_grade["true_grade"] = dataset_grade.apply(lambda x: get_true_grade(x["Correction"]), axis=1)
dataset_grade["reponse_grade_converted"] = dataset_grade.apply(lambda x: convert_rep_grade(x["Réponse GRADE"]), axis=1)


categorical_features = ['Structure', 'Size', 'Number', 'Lesion Margin', 'Lesion pedicle',
                    'Lesion fronds', 'Vascular architecture of the bladder wall', 'Microvascular architecture of the tumor']

pipeline_grade = Pipeline(
    [
        (
            "union",
            FeatureUnion(
                transformer_list=[
                    (
                        "categorical_features",
                        Pipeline(
                            [
                                ("selector", ItemSelector(key=categorical_features)),
                                ("onehot", OneHotEncoder(handle_unknown='ignore',
                                                        categories=[dataset_grade[col].unique()
                                                                                             for col in categorical_features])),
                            ]
                        ),
                    )
                ]
            ),
        ),
        ("classifier", tree.DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_split =5,
                                                   min_samples_leaf = 7)),
    ]
)

pipeline_stade = Pipeline(
    [
        (
            "union",
            FeatureUnion(
                transformer_list=[
                    (
                        "categorical_features",
                        Pipeline(
                            [
                                ("selector", ItemSelector(key=categorical_features)),
                                ("onehot", OneHotEncoder(handle_unknown='ignore',
                                                        categories=[dataset_stade[col].unique()
                                                                                             for col in categorical_features])),
                            ]
                        ),
                    )
                ]
            ),
        ),
        ("classifier", tree.DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_split =5,
                                                   min_samples_leaf = 7)),
    ]
)


df_grade = pd.concat([dataset_grade[dataset_grade["true_grade"]=="BG"],
                     dataset_grade[dataset_grade["true_grade"]=="HG"].sample(350)])
df_train_grade, df_test_grade = train_test_split(df_grade, test_size=0.3, random_state=42)

df_train_stade, df_test_stade = train_test_split(dataset_stade, test_size=0.3, random_state=42)

pipeline_grade.fit(df_train_grade, df_train_grade["reponse_grade_converted"])
pred_grade = pipeline_grade.predict(df_test_grade)

pipeline_stade.fit(df_train_stade, df_train_stade["reponse_stade_converted"])
pred_stade = pipeline_stade.predict(df_test_stade)

print(df_train_grade["true_grade"].unique())
model_grade_picked = {"model": pipeline_grade,
                      "metadata":
                          {
                              "name" : "Modele prediction grade",
                              "Author" : "fianfian",
                              "metrics": classification_report(df_test_grade["true_grade"], pred_grade, output_dict=True),
                              "required_input" : categorical_features
                          }
                      }

model_stade_picked = {"model": pipeline_stade,
                      "metadata":
                          {
                              "name" : "Modele prediction stade",
                              "Author" : "fianfian",
                              "metrics": classification_report(df_test_stade["true_stade"], pred_stade, output_dict=True),
                              "required_input" : categorical_features
                          }
                      }


# DOT data grade
dot_data_grade = tree.export_graphviz(pipeline_grade.steps[1][1], out_file =None, filled=True,
                  feature_names=pipeline_grade['union'].transformer_list[0][1]['onehot']\
                   .get_feature_names(categorical_features),
                                class_names=df_train_grade["true_grade"].unique())

# Draw graph grade
graph_grade = graphviz.Source(dot_data_grade, format="png")


# DOT data stade
dot_data_stade = tree.export_graphviz(pipeline_stade.steps[1][1], out_file =None, filled=True,
                  feature_names=pipeline_stade['union'].transformer_list[0][1]['onehot']\
                   .get_feature_names(categorical_features),
                                class_names=df_train_stade["true_stade"].unique())

# Draw graph grade
graph_stade = graphviz.Source(dot_data_stade, format="png")

graph_grade.render("../static/decision_tree_grade_final")
graph_stade.render("../static/decision_tree_stade_final")

dump(model_grade_picked, "Modeles/model_grade.joblib")
dump(model_stade_picked, "Modeles/model_stade.joblib")


