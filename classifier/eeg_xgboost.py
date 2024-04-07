from xgboost import XGBClassifier
import pandas as pd
import numpy as np

lyh_data_path = ['Baseline/lyh_data/session2/Feature_extracted/feature_extracted_se2.csv',\
    'Baseline/lyh_data/session2/Feature_extracted/feature_extracted_se2.csv']
pre_trained_model_path='/mnt/workspace/Baseline/classifier/model_to_load/xgb_4s_se1.json'
# '/mnt/workspace/Baseline/xgb_4s_se2.json'
# '/mnt/workspace/Baseline/xgb_4s_se1se2.json'

class EEG_Xgboost:
    '''
    self.trian(new_data): 使用新的数据new_data和之前lyh的数据训练, new_data: pd.Dataframe, 
    可以调用EEG_DataLoader的接口获得。
    self.predict(X_pred): 预测, X_pred: 一般sklearn的classifiers接收的格式。
    '''
    def __init__(self, old_data_path=lyh_data_path):
        self.old_data_path = old_data_path
        # self.pre_trained_model_path = pre_trained_model_path
        self.model = XGBClassifier()
        # self.model.load_model(self.pre_trained_model_path)

    def fine_tune(self, new_data):
        data = [new_data]
        for path in self.old_data_path:
            data.append(pd.read_csv(path).sample(frac=1))
        data = pd.concat(data, axis=0)
        y = data['label']
        X = data.drop(columns=['raw data', 'label'])
        self.model.fit(X, y)

    def train(data):
        y = data['label']
        X = data.drop(columns=['raw data', 'label'])
        self.model.fit(X, y)

    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        return y_pred

    def load_pretrained_model(self, path):
        self.model.load_model(path)

if __name__ == "__main__":
    eeg_xgboost = EEG_Xgboost()
    # eeg_xgboost.train()
    trail = [np.ones(shape=(140,))]
    print(eeg_xgboost.predict(trail))


