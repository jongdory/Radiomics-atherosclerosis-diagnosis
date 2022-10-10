import numpy as np
import pandas as pd
import scipy.io as sio
import pickle

from scipy import stats
from sklearn.linear_model import LogisticRegression # LR Classifier
from sklearn.svm import SVC                         # SVM Classifier
from sklearn.ensemble import RandomForestClassifier # RF Classifier
import json
import xgboost

def predict(model_dir, save_dir):
    with open(model_dir, 'rb') as f:
        model = pickle.load(f)

    labels = model.predict(RadiomicsFeatures_df)

    json_dict = {}
    for label, p_info in zip(labels,RadiomicsFeatures_info):
        case_i, art_i, anno_i = p_info.split('_')
        anno_i = str(int(anno_i))

        if case_i in json_dict:
            json_dict[case_i][art_i].update({anno_i: int(label)})    
        else:
            json_dict[case_i] = {"L":{}, "R":{}}
            json_dict[case_i][art_i].update({anno_i: int(label)})

    with open(save_dir, 'w') as fp:
        json.dump(json_dict, fp)

def predict_xgb(model, model_dir, save_dir):
    model.load_model(model_dir)

    labels = model.predict(RadiomicsFeatures_df)
    json_dict = {}
    for label, p_info in zip(labels,RadiomicsFeatures_info):
        case_i, art_i, anno_i = p_info.split('_')
        anno_i = str(int(anno_i))

        if case_i in json_dict:
            json_dict[case_i][art_i].update({anno_i: int(label)})
        else:
            json_dict[case_i] = {"L":{}, "R":{}}
            json_dict[case_i][art_i].update({anno_i: int(label)})

    with open(save_dir, 'w') as fp:
        json.dump(json_dict, fp)


if __name__ == '__main__':

    RadiomicsFeatures            = np.load("./feature/Radiomics.npz")
    RadiomicsFeatures_names      = RadiomicsFeatures['FeatureNames'].tolist()
    RadiomicsFeatures_feature    = RadiomicsFeatures['TotalFeatures']
    RadiomicsFeatures_info       = RadiomicsFeatures['p_info'].squeeze()
    RadiomicsFeatures_zscore     = stats.zscore(RadiomicsFeatures_feature)
    RadiomicsFeatures_df         = pd.DataFrame.from_dict(RadiomicsFeatures_zscore)
    RadiomicsFeatures_df.columns = RadiomicsFeatures_names

    RadiomicsFeautres_label = RadiomicsFeatures['label'].squeeze()

    xgb = xgboost.XGBClassifier(n_estimators = 700, max_depth = 8 , gamma = 0.4)

    predict("./saved_model/lr.pkl" , './results/lr.json')
    predict("./saved_model/svm.pkl", './results/svm.json')
    predict("./saved_model/rf.pkl" , './results/rf.json')
    predict_xgb(xgb, "./saved_model/xgb.json", './results/xgb.json')

    print("Results saved")
