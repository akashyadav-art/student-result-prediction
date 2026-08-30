from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    hours = float(request.form["hours"])
    attendance = float(request.form["attendance"])

    # Input
    data = [[hours, attendance]]

    # Prediction
    prediction = model.predict(data)[0]

    # Probability
    probability = model.predict_proba(data)[0][1] * 100

    if prediction == 1:
        result = "PASS ✅"
    else:
        result = "FAIL ❌"

    return render_template(
        "index.html",
        result=result,
        probability=round(probability, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)