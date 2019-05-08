import flask
from flask import request
from predictor_api import make_prediction, feature_names

# Initialize the app

app = flask.Flask(__name__)

# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!
@app.route("/")
def hello():
    return "It's alive!!!"

@app.route("/predict", methods=["GET"])
def get_predict_page():
    "Returns the rendered page"
    # We can try other template pages as well.
    # Change this line to any of the other html files in the template folder
    # Described in README
    return flask.render_template('predictor_javascript_bootstrap.html')


# Start the server, continuously listen to requests.
# We'll have a running web app!

if __name__=="__main__":
    # For local development:
    #app.run(debug=True)
    # For public web serving:
    #app.run(host='0.0.0.0')
    app.run()
