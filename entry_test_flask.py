from flask import Flask
from flask import jsonify
from flask import Response
from flask import request
import json
import linear_reg_original
from linear_reg_original import Linear_reg

app = Flask(__name__)

@app.route("/")
def root():
    print "sahil"
    response = Response(content_type="application/json")
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    print("Request data ", requestData)

if (__name__=='__main__'):
    app.run(debug=True)
