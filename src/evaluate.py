#load the train model

import joblib
import numpy as np

def evaluate_mode():

    # load the train mode
    model = joblib.load("models/model.pkl")
    test_x=np.array([[6],[7],[8]])
    #predictions
    preds = model.predict(test_x)
    print("Prediactions:",preds)
    return preds

if __name__=="__main__":
    evaluate_mode()
