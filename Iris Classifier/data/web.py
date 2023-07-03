from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html',**locals())


@app.route('/predict', methods=['POST'])
def predict():
    SL = float(request.form['SL'])
    SW = float(request.form['SW'])
    PL = float(request.form['PL'])
    PW = float(request.form['PW'])
    result_logr = logr_model.predict([[SL, SW, PL, PW]])[0]
    result_dt= dt_mod.predict([[SL, SW, PL, PW]])[0]
    result_nn= nn_model.predict([[SL, SW, PL, PW]])[0]
    result_svc= svc.predict([[SL, SW, PL, PW]])[0]
    result_linr= linr_model.predict([[SL, SW, PL, PW]])[0]
    
    return render_template('index.html', **locals())

if __name__ == '_main_':
    app.run(debug=True)