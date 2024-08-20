from cfg import sys, pd, json, np, load,LabelEncoder
sys.path.append("/core/")
from core.preprocess import predict, create_dictionary
from core.preprocess import create_features

DATASET_PATH = 'data/2cls_spam_text_cls.csv'
df = pd.read_csv (DATASET_PATH)

messages = df['Message'].values.tolist ()
labels = df['Category'].values.tolist()

if __name__ == "__main__":
    model = load('gaussian_nb_model.joblib')
    dictionary = create_dictionary(messages)

    test_input = 'I am actually thinking a way of doing something useful'
    prediction_cls = predict(test_input, model, dictionary)
    print (f'Prediction :{prediction_cls}')