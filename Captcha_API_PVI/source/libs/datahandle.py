from flask import jsonify


def failed_format(data_code, message, data, status_code):
    data = {
        "code": data_code,
        "message": message,
        "data": data,
        "status": "failed"
    }
    return jsonify(data)


def success_format(data):
    return jsonify(data)
