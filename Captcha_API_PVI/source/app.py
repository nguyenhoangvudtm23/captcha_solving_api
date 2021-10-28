import os
from flask import Flask, request
from flask_cors import CORS
from libs import config
import logging
from waitress import serve
from werkzeug.utils import secure_filename
from libs.datahandle import *
from yolov4_tiny_model import *

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

cfg = config.get_config()

validation = cfg["authen"]
# UPLOAD_FOLDER = r'.'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def validate_project(headerReq, cfg):
    project = headerReq.get("project", "")
    apikey = headerReq.get("apikey", "")

    if not project or not apikey:
        return failed_format('404.1', 'Header is missing', None, 400)
    else:
        if not any(project == c['project'] and apikey == c['apikey'] for c in cfg):
            return failed_format('401.1', 'API key or project key is not valid', None, 400)
    return True


def get_files_from_request(name=None):
    try:  # from form
        files = request.files.getlist(name)
        assert len(files) > 0
        for f in files:
            print(f)
            if str(f.filename).split('.')[1] not in ['png', 'jpg', 'PNG', 'jpeg', 'JPG']:
                yield None, request.stream
            else:
                filename = secure_filename(f.filename)
                f.save(os.path.join('image.png'))
                yield f.filename, f.stream
    except Exception:  # from bytes
        yield None, request.stream


@app.route("/healthz", methods=["GET"])
def healthz():
    return "Welcome to Flask Example Service"


@app.route("/extraction", methods=["POST"])
def extract():
    for file, buffer in get_files_from_request("file"):
        if not validate_project(request.headers, validation) is True:
            return validate_project(request.headers, validation)
        image_path = os.path.join(os.path.dirname(__file__), 'image.png')
        buffer = buffer.read()
        result = {
            "name": str(file),
            "size": len(buffer),
            "result": detect_image(image_path)
        }
        return success_format(result)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8500
    logging.debug("Run Menu at http://" + str(host) + ":" + str(port) + '/')

    serve(app, host=host, port=port, threads=8)

