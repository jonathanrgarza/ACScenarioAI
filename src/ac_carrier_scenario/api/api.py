from typing import Optional

from flask import Flask, request, Response, make_response

from ac_carrier_scenario.api.helpers import get_scenario_from_json, perform_analysis, get_flask_response, \
    get_performance_stats
from ac_carrier_scenario.common.scenarios import AircraftCarrierScenario

app = Flask(__name__)


@app.route("/")
def index() -> Response:
    response = make_response("Make sure to use the API endpoints.", 200)
    response.mimetype = "text/plain"
    return response


@app.route("/api/analysis", methods=["POST"])
def analysis() -> Response:
    content_json: Optional[dict] = request.get_json()

    is_valid_request: bool = False
    is_valid_scenario: bool = False

    results: Optional[dict] = None

    if content_json is not None:
        is_valid_request = True
        # Turn request into a scenario

        scenario: Optional[AircraftCarrierScenario] = get_scenario_from_json(content_json)

        if scenario is not None:
            is_valid_scenario = True
            # Perform analysis
            results = perform_analysis(scenario)

    # Make response
    response = get_flask_response(is_valid_request, is_valid_scenario, results)
    return response


@app.route("/api/performance", methods=["POST"])
def performance() -> Response:
    content_json: Optional[dict] = request.get_json()

    is_valid_request: bool = False
    is_valid_scenario: bool = False

    results: Optional[dict] = None

    if content_json is not None:
        is_valid_request = True
        # Turn request into a scenario

        scenario: Optional[AircraftCarrierScenario] = get_scenario_from_json(content_json)

        if scenario is not None:
            is_valid_scenario = True
            # Perform analysis
            results = get_performance_stats(scenario)

    # Make response
    response = get_flask_response(is_valid_request, is_valid_scenario, results)
    return response


if __name__ == "__main__":
    app.run()
