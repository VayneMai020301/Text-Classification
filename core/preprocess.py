from cfg import nltk, string, np, json, joblib
from cfg import train_test_split, LabelEncoder
from cfg import VAL_SIZE, TEST_SIZE, SEED

def lowercase (text) :
    return text.lower ()

def punctuation_removal(text ) :
    translator = str.maketrans ("", "", string.punctuation )    
    return text.translate ( translator )

def tokenize( text ) :
    return nltk.word_tokenize ( text )

def remove_stopwords(tokens) :
    stop_words = nltk.corpus.stopwords.words('english')
    return [token for token in tokens if token not in stop_words ]

def stemming (tokens) :
    stemmer = nltk.PorterStemmer ()
    return [stemmer.stem(token) for token in tokens ]

def preprocess_text ( text ) :
    text = lowercase ( text )
    text = punctuation_removal ( text )
    tokens = tokenize ( text )
    tokens = remove_stopwords ( tokens )
    tokens = stemming ( tokens )

    return tokens

def preprocess_text ( text ) :
    text = lowercase ( text )
    text = punctuation_removal ( text )
    tokens = tokenize ( text )
    tokens = remove_stopwords ( tokens )
    tokens = stemming ( tokens )

    return tokens


def create_dictionary ( messages ) :
    dictionary = [] 
    for tokens in messages :
        for token in tokens :
            if token not in dictionary :
                dictionary.append(token)   
    return dictionary


def create_features (tokens , dictionary ) :
    features = np. zeros ( len ( dictionary ) ) 
    for token in tokens :
        if token in dictionary :
            features [ dictionary . index ( token ) ] += 1  
    return features

def data_prepatation(messages,labels):

    dictionary = create_dictionary(messages)
    X = np.array([create_features(tokens,dictionary ) for tokens in messages])
    le = LabelEncoder ()
    y = le.fit_transform(labels)
    joblib.dump(le, 'label_encoder.joblib')
    x_train , x_val , y_train , y_val = train_test_split (X, y,
                                                            test_size = VAL_SIZE ,
                                                            shuffle =True ,
                                                            random_state = SEED )

    x_train , x_test , y_train , y_test = train_test_split ( x_train , y_train ,
                                                            test_size = TEST_SIZE ,
                                                            shuffle =True ,
                                                            random_state = SEED )

    x_train = x_train.tolist()  
    x_val = x_val.tolist()
    y_train = y_train.tolist()
    y_val = y_val.tolist()
    x_test = x_test.tolist()
    y_test = y_test.tolist()

    data = {
        'X_train': x_train,
        'X_val': x_val,
        'y_train': y_train,
        'y_val': y_val,
        'X_test': x_test,
        'y_test': y_test
    }

    with open('dataset.json', 'w') as f:
        json.dump(data, f)

    return  x_train , x_val , y_train , y_val, x_train , x_test , y_train , y_test


def predict (text , model , dictionary ) :
    le = joblib.load('label_encoder.joblib')
    features = create_features(text,dictionary )
    features = np. array(features).reshape (1 , -1)
    prediction = model.predict(features )
    prediction_cls = le.inverse_transform(prediction )[0]   
    return prediction_cls

