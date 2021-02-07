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

__locations = [x.upper() for x in __locations]

@app.route('/',methods=['GET'])
def home():
        return render_template('index.html',areas = __locations)


@app.route('/predict',methods=['POST'])
def predict():
        #postedBy = int(request.form['postedby'])
        underconstruction = int(request.form['uiuc'])
        rera  = int(request.form['uirera'])
        bhk = int(request.form['uiBHK'])
        sqft = float(request.form['Squareft'])
        readytomove = int(request.form['uirtm'])
        resale = int(request.form['uiresale'])
        location = request.form['area']
        
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

        output = round(__model.predict([x])[0], 2)

        return render_template('index.html', areas = __locations,prediction_text='{} lakh'.format(output))




if __name__ == '__main__':
    app.run(debug=True)   


