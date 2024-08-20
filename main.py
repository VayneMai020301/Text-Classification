import sys
import pandas as pd
from joblib import dump

sys.path.append("/core/")
from cfg import GaussianNB

from core.preprocess import data_prepatation

DATASET_PATH = 'data/2cls_spam_text_cls.csv'
df = pd.read_csv (DATASET_PATH)

messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()

if __name__ == "__main__":
    x_train , x_val , y_train , y_val, x_train , x_test , y_train , y_test= data_prepatation(messages,labels) 
    model = GaussianNB()
    print('Start training ... ')
    model = model.fit(x_train , y_train )
    dump(model, 'gaussian_nb_model.joblib')
    print('Training completed !')