from flask import Flask
from flask import jsonify
from flask import Response
from flask import request
import json
import linear_reg_original
import pearson_corr_original
import chi_sqr_original

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])

def root():
    response = Response(content_type="application/json")
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    print("Request data ", requestData)


    hdfsLocation = ''

    responseData = {}

    fileLocation = requestData.get("fileLocation")
    print("file Location", fileLocation)
    feature_colm_req = requestData.get("features_column")
    print("features colm", feature_colm_req)
    label_colm_req = requestData.get("label_column")
    print("label colm ", label_colm_req)
    algo_name = requestData.get("algorithm_name")
    print ("algo name ", algo_name)

    responseData = ''

    try:
        if algo_name == "linear_reg":
            responseData = linear_reg_original.Linear_reg(dataset_add=fileLocation, feature_colm=feature_colm_req, label_colm=label_colm_req)
        elif algo_name == 'pearson_test':
            responseData = pearson_corr_original.Correlation(dataset_add=fileLocation, feature_colm=feature_colm_req,label_colm=label_colm_req)
        elif algo_name == 'chi_square_test':
            responseData = chi_sqr_original.Chi_sqr(dataset_add=fileLocation, feature_colm=feature_colm_req, label_colm=label_colm_req)

    except Exception as e:
        print ('exception is = ' + str(e))
        responseData = str(json.dumps({'run_status ' : 'request not processed '})).encode('utf-8')


    # return iter([responseData])

    print(responseData)

    return jsonify(success='success', message = 'it was a success',data= responseData)

if (__name__=='__main__'):

    app.run(host='10.171.0.173', port = 3333, debug=False)


