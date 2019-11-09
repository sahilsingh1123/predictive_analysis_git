#this class is designed for handling exception on various errors

import json

class PredictiveExceptionHandling():

    defaultException =''
    exceptionMessage=''
    #list of exceptions
    #'Unable to infer schema for Parquet. It must be specified manually.;'
    #'requirement failed: The input column indexed_Exhaust_Gas_Bypass_Valve_Position should have at least two distinct values.'

    def exceptionHandling(exception):
        defaultException = str(exception)
        if (defaultException.startswith("'Unable to infer schema for Parquet")):
            exceptionMessage = "Unable to infer schema for Parquet"

        elif (defaultException.endswith("should have at least two distinct values.'")):
            defaultException = defaultException.replace("indexed_", "", 1)
            exceptionMessage = defaultException

        else:
            exceptionMessage = defaultException

        responseData = {'run_status': exceptionMessage}

        return responseData
