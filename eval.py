import numpy as np
import pandas as pd
import scipy.io as sio
import pickle

from scipy import stats
from sklearn.linear_model import LogisticRegression # LR Classifier
from sklearn.svm import SVC                         # SVM Classifier
from sklearn.ensemble import RandomForestClassifier # RF Classifier
import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report   # metrics (sensitivity, specificity, f1-score)
from sklearn.metrics import r2_score                # R-squared
from sklearn.metrics import roc_auc_score           # roc_auc curve
from sklearn.metrics import roc_curve
import json
import xgboost
import warnings

warnings.filterwarnings('ignore')

def print_metric(model, modelname, X, y):
    
    target_names = ['Normal', 'lession']
    
    predict      = model.predict(X)
    predict_prob = model.predict_proba(X)[:,1]
    
    fper, tper, th = roc_curve(y, predict_prob)
    
    metric = classification_report(y, predict, target_names=target_names, output_dict=True)
    
    # AUC
    AUC = (roc_auc_score(y, predict))
    
    # adjusted R-squared
    adjC = (len(y)-1)/(len(y)-X.shape[1]-1)
    AR2  = 1 - (1-r2_score(y, predict_prob))*adjC
    
    # cohen's kappa
    CK   = sklearn.metrics.cohen_kappa_score(y, predict)
    
    print(f"--------------{modelname} Performance--------------\n")
    print(f" Accuracy   :" , metric['accuracy'])
    print(f" Sensitivity:" , metric['Normal']['recall'])
    print(f" Specificity:" , metric['lession']['recall'])
    print(f" AUC        :" , AUC)
    print(f" Adjusted R2:" , AR2)
    print(f" cohensKappa:" , CK)
    print("\n")


if __name__ == "__main__":

    RadiomicsFeatures            = np.load("./feature/Radiomics.npz")
    RadiomicsFeatures_names      = RadiomicsFeatures['FeatureNames'].tolist()
    RadiomicsFeatures_feature    = RadiomicsFeatures['TotalFeatures'].astype(float)
    RadiomicsFeatures_info       = RadiomicsFeatures['p_info'].squeeze()
    RadiomicsFeatures_zscore = stats.zscore(RadiomicsFeatures_feature)
    RadiomicsFeatures_df = pd.DataFrame.from_dict(RadiomicsFeatures_zscore)
    RadiomicsFeatures_df.columns = RadiomicsFeatures_names

    RadiomicsFeatures_label = RadiomicsFeatures['label'].squeeze()

    lr  = LogisticRegression(C=4.5, 
                                penalty="l1", 
                                solver='liblinear', 
                                max_iter=3000,
                                class_weight={0:1 , 1:1.25})
    svm = SVC(gamma='auto', probability=True)
    rf  = RandomForestClassifier()
    xgb = xgboost.XGBClassifier(n_estimators=700, max_depth = 8 , gamma = 0.4)

    with open("./saved_model/lr.pkl", 'rb') as file:
        lr = pickle.load(file)

    with open("./saved_model/svm.pkl", 'rb') as file:
        svm = pickle.load(file)

    with open("./saved_model/rf.pkl", 'rb') as file:
        rf = pickle.load(file)

    xgb.load_model("./saved_model/xgb.json")

    print_metric(lr , "lr" , RadiomicsFeatures_df, RadiomicsFeatures_label)
    print_metric(svm, "svm", RadiomicsFeatures_df, RadiomicsFeatures_label)
    print_metric(rf , "rf" , RadiomicsFeatures_df, RadiomicsFeatures_label)
    print_metric(xgb, "xgb", RadiomicsFeatures_df, RadiomicsFeatures_label)