from cfg import accuracy_score, load
from cfg import json, np

if __name__ == "__main__":
    model = load('gaussian_nb_model.joblib')    

    with open('dataset.json', 'r') as f:
        data = json.load(f)

    x_val = np.array(data['X_val'])
    y_val = np.array(data['y_val'])
    x_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])

    print('Start evaluatation ... ')
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test )

    val_accuracy = accuracy_score(y_val , y_val_pred )
    test_accuracy = accuracy_score(y_test , y_test_pred )

    print (f'Val accuracy : { val_accuracy }')
    print (f'Test accuracy : { test_accuracy }')