from flask import Flask
from flask import request
import numpy as np
import pickle


with open('./banglore_home_prices_model.pickle', 'rb') as p:
    pipe = pickle.load(p)    

app = Flask(__name__)

@app.route('/pricePrediction', methods=['POST'])
def index():
    temp=np.zeros(244)
    temp[int(request.form["location"])]=1
    temp[0] = request.form["sqft"]
    temp[1] = request.form["bath"]
    temp[2] = request.form["bhk"]
    price = pipe.predict([temp])[0]
    price=price.item()
    price=round(price,2)
    priceDict = {"price" : price}
    return priceDict

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
