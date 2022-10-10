import numpy as np
import pandas as pd
import scipy.io as sio
import pickle

from scipy import stats
from sklearn.linear_model import LogisticRegression # LR Classifier
from sklearn.svm import SVC                         # SVM Classifier
from sklearn.ensemble import RandomForestClassifier # RF Classifier
import xgboost

if __name__ == '__main__':
    RadiomicsFeatures = np.load("./feature/Radiomics.npz")
    RadiomicsFeatures_names = RadiomicsFeatures['FeatureNames'].tolist()
    RadiomicsFeatures_feature = RadiomicsFeatures['TotalFeatures']
    print(len(RadiomicsFeatures_names))
    print(RadiomicsFeatures_feature.shape)
    RadiomicsFeatures_zscore = stats.zscore(RadiomicsFeatures_feature)
    RadiomicsFeatures_df = pd.DataFrame.from_dict(RadiomicsFeatures_zscore)
    RadiomicsFeatures_df.columns = RadiomicsFeatures_names

    RadiomicsFeatures_label = RadiomicsFeatures['label'].squeeze()

    # models
    lr  = LogisticRegression(C=4.5, 
                                penalty="l1", 
                                solver='liblinear', 
                                max_iter=3000,
                                class_weight={0:1 , 1:1.25})
    svm = SVC(gamma='auto', probability=True)
    rf  = RandomForestClassifier()
    xgb = xgboost.XGBClassifier(n_estimators = 700, max_depth = 8 , gamma = 0.4)

    # train model
    lr.fit( RadiomicsFeatures_df,RadiomicsFeatures_label)
    svm.fit(RadiomicsFeatures_df,RadiomicsFeatures_label)
    rf.fit( RadiomicsFeatures_df,RadiomicsFeatures_label)
    xgb.fit(RadiomicsFeatures_df,RadiomicsFeatures_label)

    # save model
    with open("./saved_model/lr.pkl",'wb') as f:
        pickle.dump(lr,f)
    with open('./saved_model/svm.pkl','wb') as f:
        pickle.dump(svm,f)
    with open('./saved_model/rf.pkl','wb') as f:
        pickle.dump(rf,f)
    xgb.save_model("./saved_model/xgb.json")

    print("Models saved")