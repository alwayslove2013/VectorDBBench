from flask import Flask, jsonify
from vectordb_bench.interface import benchMarkRunner
import json

app = Flask(__name__)


@app.route("/get_res", methods=["GET"])
def get_res():
    """task label -> res"""
    allResults = benchMarkRunner.get_results()
    a = allResults[0]

    return jsonify(json.loads(a.json()))


def get_status():
    "running 5/18, not running"
    return {}


def stop():
    return {}


def run():
    return {}


def main():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
