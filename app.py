from flask import Flask, render_template, request
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read all form values as floats
        features = [float(x) for x in request.form.values()]
        
        # Make prediction
        prediction = model.predict([features])[0]

        # Prepare a user-friendly result
        if prediction == 1:
            result = "⚠️ Patient is at HIGH risk of heart failure. Please consult a doctor!"
        else:
            result = "✅ Patient is at LOW risk of heart failure. Keep up the healthy habits!"
        
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
