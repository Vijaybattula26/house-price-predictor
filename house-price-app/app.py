from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric fields
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = float(request.form['housing_median_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])

        # Categorical field
        ocean_proximity = request.form['ocean_proximity']

        # One-hot encode ocean proximity
        ocean_dict = {
            'NEAR BAY': [1, 0, 0, 0],
            'NEAR OCEAN': [0, 1, 0, 0],
            'INLAND': [0, 0, 1, 0],
            'ISLAND': [0, 0, 0, 1],
            'UNKNOWN': [0, 0, 0, 0]  # fallback
        }
        ocean_features = ocean_dict.get(ocean_proximity, [0, 0, 0, 0])

        final_features = np.array([longitude, latitude, housing_median_age, total_rooms,
                                   total_bedrooms, population, households, median_income] + ocean_features).reshape(1, -1)

        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
