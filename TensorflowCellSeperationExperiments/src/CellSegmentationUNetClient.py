import base64
import requests

import numpy as np

def run(input_mask):
    mask = np.ravel(input_mask.copy()).astype(np.float32)
    bts = input_mask.tobytes()
    data = base64.urlsafe_b64encode(bts)
    response = requests.post("http://127.0.0.1/run",data={"input_mask_data":data})
    output_data_bts =base64.urlsafe_b64decode(response.json()["output_mask_data"])
    output_mask = np.frombuffer(output_data_bts,dtype=np.float32).reshape((128,128))
    return output_mask