from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('heart.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('heart.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = int(prediction[0])

    if output == 0:
        return render_template('heart.html', prediction_text= 'Heart Disease : No')
    else:
        return render_template('heart.html', prediction_text= 'Heart Disease : Yes') 


if __name__ == "__main__":
    app.run(debug=True)