import requests

# URL of the 'web_prediction' endpoint
url = "http://127.0.0.1:8005/web_prediction"

# Data to be sent to the server (sample input values)
data = {
    "Pregnancies": 4,
    "Glucose": 120,
    "BloodPressure": 80,
    "SkinThickness": 20,
    "Insulin": 70,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}

# Send POST request with the input data
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Get and print the prediction result
    prediction = response.json()
    print(f"Prediction: {prediction}")
else:
    print(f"Request failed with status code {response.status_code}")
