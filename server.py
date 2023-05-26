from flask import Flask, request, jsonify, render_template
import util
app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    result = util.create_future_df(start_date, end_date)

    # Return the result as a JSON object
    return render_template("index.html", prediction_text = "Estimated electricity usage between {} and {} is {}MW".format(start_date, end_date, result))

# @app.route("/predict_electrcity_consumption", methods = ["POST"])
# #   DateTIme object

if __name__ == "__main__":
    print("Staritng python Flask server for electricity")
    app.run()