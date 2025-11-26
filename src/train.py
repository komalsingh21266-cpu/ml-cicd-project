import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os


# create a simple dataset
def train_model():
    
    data = pd.DataFrame({
        'x':[1,2,3,4,5],
        'y':[2, 2, 4, 2,6]
})

    x = data[['x']]
    y = data['y']

# train the liner regression model
    model = LinearRegression()
    model.fit(x,y)

#save
    os.makedirs("models",exist_ok=True)
    joblib.dump(model,"models/model.pkl")
    print("model trained ans saved")

if __name__=="__main__":
    train_model()