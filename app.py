from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

global __locations
global __data_columns

__locations = None
__data_columns = None
__model = None

with open("./models/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[6:]


with open('./models/pune_house_prices_model.pkl', 'rb') as f:
        __model = pickle.load(f)        


@app.route('/',methods=['GET'])
def home():
    loc_list = [x.upper() for x in __locations]
    return render_template('index.html',areas = loc_list)


@app.route('/predict_house_price',methods=['POST','GET'])
def predict_house_price():
        #postedBy = int(request.form['postedby'])
        underconstruction = int(request.form['underConstruction'])
        rera  = int(request.form['rera'])
        bhk = int(request.form['bhk'])
        sqft = float(request.form['total_sqft'])
        readytomove = int(request.form['readytoMove'])
        resale = int(request.form['resale'])
        location = request.form['location']
        
        try:
            loc_index = __data_columns.index(location.lower())
        except:
            loc_index = -1

        x = np.zeros(len(__data_columns))
        x[0] = underconstruction
        x[1] = rera
        x[2] = bhk
        x[3] = sqft
        x[4] = readytomove
        x[5] = resale

        if loc_index >= 0:
            x[loc_index] = 1

        response = jsonify({
            'estimated_price': round(__model.predict([x])[0], 2)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response




if __name__ == '__main__':
    app.run(debug=True)   


