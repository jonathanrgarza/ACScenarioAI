from typing import Optional

from flask import Flask, request, Response, jsonify, make_response
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from ac_carrier_scenario.scenarios import AircraftCarrierScenario

app = Flask(__name__)


@app.route("/")
def index() -> Response:
    response = make_response("Make sure to use the API endpoints.", 200)
    response.mimetype = "text/plain"
    return response


@app.route("/api/analysis", methods="POST")
def analysis() -> Response:
    content: Optional[dict] = request.get_json()

    is_valid_request: bool = False
    is_valid_scenario: bool = False

    if content is not None:
        is_valid_request = True
        # Turn request into a scenario


        scenario: AircraftCarrierScenario =

    # Make response

    if not is_valid_request:
        response = make_response("{error:\"No JSON data was submitted or mimetype was not 'application/json'\"}",
                                 400)  # Bad Request
        response.mimetype = "application/json"
    elif not is_valid_scenario:
        response = make_response("{error:\"Not a valid scenario\"}",
                                 400)  # Bad Request
        response.mimetype = "application/json"
    else:
        response = make_response("DATA", 200)
        response.mimetype = "application/json"

    return response


@app.route("/api/performance", methods="POST")
def performance() -> Response:

    # Make response
    response = make_response("DATA", 200)
    response.mimetype = "application/json"
    return response


if __name__ == "__main__":
    app.run()
