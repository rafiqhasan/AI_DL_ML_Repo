import base64
import os
from grpc.beta import implementations
import tensorflow as tf
#import skimage.io
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib import predictor
import json
import pandas as pd
import requests
from flask import Flask, render_template, send_from_directory, globals, request, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rn-0n75(s^3ku^qrnafm*ak$lb_=jncll_)_ol^thqhq8u+88q'
socketio = SocketIO(app)

#service = (json.loads(os.getenv('VCAP_SERVICES', '')))['ml-foundation'][0]

MODEL_NAME = str(os.getenv('MODEL_NAME', 'grir'))
client_id = "sb-143e9623-3721-4f82-9673-662ade12be0a!b7026|foundation-std-mlftrial!b3410" #str(service['credentials']['clientid'])
client_secret = "dGxQcU/lF2sK3YjzvsBI5kausmI=" #str(service['credentials']['clientsecret'])
authentication_url = "https://p1161101trial.authentication.eu10.hana.ondemand.com/oauth/token" #str(service['credentials']['url']) + "/oauth/token"

def cml_get_predictions(data_req,model="grir_ml",ver="v1"):
    name = 'projects/{}/models/{}'.format('sap-dev-sbox', model)
    name += '/versions/{}'.format(ver)

    data = data = request.get_json(force=True)['DATA']
    json_body_ml = {'instances': data}

def get_access_token():
    #var = 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImtleS1pZC0xIn0.eyJqdGkiOiJjNGNjYWNhODg4MTA0ZGQ5YjE1NDJkMDkzMTJlYzhmNCIsImV4dF9hdHRyIjp7ImVuaGFuY2VyIjoiWFNVQUEiLCJ6ZG4iOiJwMTE2MTEwMXRyaWFsIiwic2VydmljZWluc3RhbmNlaWQiOiIxNDNlOTYyMy0zNzIxLTRmODItOTY3My02NjJhZGUxMmJlMGEifSwic3ViIjoic2ItMTQzZTk2MjMtMzcyMS00ZjgyLTk2NzMtNjYyYWRlMTJiZTBhIWI3MDI2fGZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwIiwiYXV0aG9yaXRpZXMiOlsiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAubW9kZWxyZXBvLnJlYWQiLCJ1YWEucmVzb3VyY2UiLCJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5yZXRyYWluc2VydmljZS53cml0ZSIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLmZ1bmN0aW9uYWxzZXJ2aWNlLmFsbCIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLnJldHJhaW5zZXJ2aWNlLnJlYWQiLCJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5tb2RlbHNlcnZpY2UucmVhZCIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLm1vZGVsbWV0ZXJpbmcucmVhZCIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLmRhdGFtYW5hZ2VtZW50LndyaXRlIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAubW9kZWxyZXBvLndyaXRlIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAuZGF0YW1hbmFnZW1lbnQucmVhZCIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLm1vZGVsZGVwbG95bWVudC5hbGwiLCJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5zdG9yYWdlYXBpLmFsbCJdLCJzY29wZSI6WyJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5tb2RlbHJlcG8ucmVhZCIsInVhYS5yZXNvdXJjZSIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLnJldHJhaW5zZXJ2aWNlLndyaXRlIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAuZnVuY3Rpb25hbHNlcnZpY2UuYWxsIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAucmV0cmFpbnNlcnZpY2UucmVhZCIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLm1vZGVsc2VydmljZS5yZWFkIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAubW9kZWxtZXRlcmluZy5yZWFkIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAuZGF0YW1hbmFnZW1lbnQud3JpdGUiLCJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5tb2RlbHJlcG8ud3JpdGUiLCJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5kYXRhbWFuYWdlbWVudC5yZWFkIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAubW9kZWxkZXBsb3ltZW50LmFsbCIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLnN0b3JhZ2VhcGkuYWxsIl0sImNsaWVudF9pZCI6InNiLTE0M2U5NjIzLTM3MjEtNGY4Mi05NjczLTY2MmFkZTEyYmUwYSFiNzAyNnxmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMCIsImNpZCI6InNiLTE0M2U5NjIzLTM3MjEtNGY4Mi05NjczLTY2MmFkZTEyYmUwYSFiNzAyNnxmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMCIsImF6cCI6InNiLTE0M2U5NjIzLTM3MjEtNGY4Mi05NjczLTY2MmFkZTEyYmUwYSFiNzAyNnxmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMCIsImdyYW50X3R5cGUiOiJjbGllbnRfY3JlZGVudGlhbHMiLCJyZXZfc2lnIjoiZWU4MGJjZTkiLCJpYXQiOjE1Mzk1MTI1ODMsImV4cCI6MTUzOTU1NTc4MywiaXNzIjoiaHR0cDovL3AxMTYxMTAxdHJpYWwubG9jYWxob3N0OjgwODAvdWFhL29hdXRoL3Rva2VuIiwiemlkIjoiZTRhYWU3NmYtYThmOC00OWYwLWIyMjctOTgxYzI2MmFkN2FlIiwiYXVkIjpbInNiLTE0M2U5NjIzLTM3MjEtNGY4Mi05NjczLTY2MmFkZTEyYmUwYSFiNzAyNnxmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMCIsInVhYSIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLnJldHJhaW5zZXJ2aWNlIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAubW9kZWxyZXBvIiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAubW9kZWxkZXBsb3ltZW50IiwiZm91bmRhdGlvbi1zdGQtbWxmdHJpYWwhYjM0MTAuZnVuY3Rpb25hbHNlcnZpY2UiLCJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5kYXRhbWFuYWdlbWVudCIsImZvdW5kYXRpb24tc3RkLW1sZnRyaWFsIWIzNDEwLnN0b3JhZ2VhcGkiLCJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5tb2RlbHNlcnZpY2UiLCJmb3VuZGF0aW9uLXN0ZC1tbGZ0cmlhbCFiMzQxMC5tb2RlbG1ldGVyaW5nIl19.AC9TQGtkA3EF_X62k3frXmKMMBcrYYoPnHu2yahF-WJdcweBz8D9iL2Jk7v7Gn-SBx7H0vQfxOzEp4kum7R75LFyS5ZgfsbgBklLf71nM2r0lL8g36YuhneLGojlWtIOlJKLA7hD1BJlGxzKGJBrsJDpPZF85dUPEk9jynUQhZII9i7g_yvXeqM6UPiXv7GJMEyeGf3lBbMFAda9er0l0A-vOLwM68YSR6E2PC9JLWZykx6kip5qnTseIsmJVwdeRw5koZP3Kf1cTpZHfhUUEmmWBpXGk4hetWgWti1mn3zMiiXC6i9jkMJ4ZHee4x3JfHEuaBnaIzSPnDwu9YoGdvHlrmGcytOrWwVZeYAOY4nkOisDZ0peDrUloU3YJrXPXIIBJmoxBoM9eMXkRAG1hKmI7GlntOHlcfbyGmOxglqoI9N-5z-ZEi_3lORypd4g3pwvk6j_Ffl75jlHWIdv8CVJzXrOvexeV28UIQYXhD4CAz-paivL_p-JXPc3L4uGvTxpHQ7gbSs4vzUN-9AH3IoMFF0QHFETMxlXWoP8Ki7IJ-HPN3DzzarXDwqHYJbGgblacoSnjFzMzSkK_Jq_QYBhNEpuCz4tS6FzlmuNgsgTwS-KyUZ5ahL4jL-T250hMUM9_z9I6XBLcdfvOLkvexBkLwMcfJ1CFxyCFqHw2Bo'

    querystring = {"grant_type": "client_credentials"}
    auth_text = client_id + ":" + client_secret
    # auth = b64encode(b"" + auth_text.encode()).decode("ascii")
    auth = base64.b64encode( bytes(auth_text, "utf-8") ).decode("ascii")
    headers = {
        'Cache-Control': "no-cache",
        'Authorization': "Basic %s" % auth
    }
    response = requests.request("GET", authentication_url, headers=headers, params=querystring)
    return 'Bearer ' + json.loads(response.text)['access_token']
    # return var

def metadata_transformer(metadata):
    additions = []
    token = get_access_token()
    additions.append(('authorization', token))
    return tuple(metadata) + tuple(additions)

def get_predictions_from_disk(request):
    #Load saved model
    export_dir = os.getcwd() + "/tflow_grir_model"
    predict_fn = predictor.from_saved_model(export_dir,signature_def_key='predict')

    #Get data from JSON
    data = request.get_json(force=True)['DATA']
    df_req = pd.io.json.json_normalize(data)
    json_body_ml = {}
    for k in df_req.columns:
        json_body_ml[k] = list(df_req[k].values)
    #json_body_ml = {'instances': data}
    #print(json_body_ml)
    # predictions = predict_fn({"DIFGRIRV": [-38100],"NODLIR": [90],"VSTATU": ["1"],"NODLGR": [0],"DIFGRIRD": [-80],"VPATD": [30],
    #                           "WERKS": ["ML01"],
    #                           "EKORG": ["1"],"TOTGRQTY": [0],"SCENARIO": ["3"],"TOTIRQTY": [80],"KTOKK": ["1"],"EKGRP": ["A"]})
    predictions = predict_fn(json_body_ml)
    out_list = []
    for i in list(predictions['probabilities'][:,1]):
        out = {}
        out['probability'] = str(i)
        out_list.append(out)
    print(out_list)
    return(out_list)

@app.route('/api', methods=['POST'])
def api():
    prob = get_predictions_from_disk(request)
    return(jsonify(prob))

@app.route('/do_inference', methods=['POST'])
def main():
    deployment_url = "https://mlftrial-deployment-api.cfapps.eu10.hana.ondemand.com" + "/api/v2/modelServers"
    querystring = {"modelName": MODEL_NAME}
    headers = {
        'Authorization': get_access_token(),
        'Cache-Control': "no-cache"
    }
    response = requests.request("GET", deployment_url, headers=headers, params=querystring)
    #print(response.text)
    model_info = json.loads(response.text)
    latest_version = [0, 0]
    for index, model in enumerate(model_info["modelServers"]):
        if int(model["specs"]["models"][0]["modelVersion"]) > latest_version[0]:
            latest_version = [int(model["specs"]["models"][0]["modelVersion"]), index]
    model_host = model_info["modelServers"][latest_version[1]]["endpoints"][0]
    credentials = implementations.ssl_channel_credentials(root_certificates=str(model_host["caCrt"]))
    channel = implementations.secure_channel(str(model_host["host"]),
                                             int(model_host["port"]), credentials)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)#, metadata_transformer=metadata_transformer)
    #uploaded_files = globals.request.files.getlist('file')
    #data = skimage.io.imread(uploaded_files[0])

    feature_dict = {
                        'DIFGRIRV': tf.train.Feature(int64_list=tf.train.Int64List(value=[-38100])),
                        'NODLIR': tf.train.Feature(int64_list=tf.train.Int64List(value=[90])),
                        'VSTATU': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"1"])),
                        'NODLGR': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                        'DIFGRIRD': tf.train.Feature(int64_list=tf.train.Int64List(value=[-80])),
                        'VPATD': tf.train.Feature(int64_list=tf.train.Int64List(value=[30])),
                        'WERKS': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"ML01"])),
                        'EKORG': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"1"])),
                        'TOTGRQTY': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                        'SCENARIO': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"3"])),
                        'TOTIRQTY': tf.train.Feature(int64_list=tf.train.Int64List(value=[80])),
                        'KTOKK': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"1"])),
                        'EKGRP': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"A"])),
                   }
    example= tf.train.Example(features=tf.train.Features(feature=feature_dict))
    data = example.SerializeToString()
    # data = {"DIFGRIRV": [-38100],"NODLIR": [90],"VSTATU": ["1"],"NODLGR": [0],"DIFGRIRD": [-80],"VPATD": [30],
    #           "WERKS": ["ML01"],
    #           "EKORG": ["1"],"TOTGRQTY": [0],"SCENARIO": ["3"],"TOTIRQTY": [80],"KTOKK": ["1"],"EKGRP": ["A"]}

    req = predict_pb2.PredictRequest()
    req.model_spec.name = MODEL_NAME
    #req.model_spec.signature_name = 'predict_images'
    req.inputs["inputs"].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1, 1]))
    res = str(stub.Predict(req, 150))
    # res = str(stub.Predict(req, 150)).split('}')[3].split('\n')
    print(res)
    res.pop(11)
    res.pop(0)
    out_val = 0.0
    out = 0
    for i, estimate in enumerate(res):
        if float(estimate[14:]) > out_val:
            out_val = float(estimate[14:])
            out = i
    return "Result: " + str(out)


@app.route('/', methods=["GET"])
def serve_index():
    return render_template('index.html')


@app.route('/<path:path>')
def static_proxy(path):
    if ".js" in path or "i18n" in path or "favicon" in path or ".json" in path or ".css" in path:
        return send_from_directory('templates', path)
    else:
        return render_template(path)


port = os.getenv('PORT', 5000)
if __name__ == '__main__':
    app.debug = not os.getenv('PORT')
    socketio.run(app, host='0.0.0.0', port=int(port))
