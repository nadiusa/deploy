import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from flask import Flask
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import json

app = Flask(__name__)


def predict():
    path = 'https://raw.githubusercontent.com/nadiusa/FIA/main/linear_regression/apartmentComplexData.csv'
    price_data = pd.read_csv(path)

    # define the x & y data.
    medianCompexValue = price_data['medianCompexValue']
    complexAge = price_data['complexAge']
    totalRooms = price_data['totalRooms']
    totalBedrooms = price_data['totalBedrooms']
    complexInhabitants = price_data['complexInhabitants']
    apartmentsNr = price_data['apartmentsNr']


    # multiple linear regression in sklearn
    X = price_data[['column1','column2','complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr','column8']]
    Y = price_data[['medianCompexValue']]
    
    # Split X and y into X_
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=50)
    
    # create a Linear Regression model object.
    regression_model = LinearRegression()
    
    # pass through the X_train & y_train data set.
    regression_model.fit(X_train, y_train)

    y_predict = regression_model.predict(X_test)
    model_mse = mean_squared_error(y_test, y_predict)
    
    # calculate the mean absolute error.
    model_mae = mean_absolute_error(y_test, y_predict)
    
    # calulcate the root mean squared error
    model_rmse =  math.sqrt(model_mse)

    model_r2 = r2_score(y_test, y_predict)
    
    return y_predict[:5], model_mse, model_mae, model_rmse, model_r2, path


@app.route("/")
def index():
    return json.dumps({"Dataset": predict()[5]}, sort_keys=True)


@app.route("/multipleprediction")
def multiple_prediction():
    predict_result = predict()
    return json.dumps({"Multiple Prediction": str(predict_result[0])})


@app.route('/errors')
def errors():
    predict_result = predict()
    return json.dumps({"MSE": str(predict_result[1]), "MAE": str(predict_result[2]), "RMSE": str(predict_result[3]), "R2": str(predict_result[4])})


if __name__ == '__main__':
    app.run()