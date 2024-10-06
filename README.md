# AC Carrier Scenario

## Description
This project was developed as part of a research project to develop an AI that can find ideal use of resources against a set of targets.
The AI agent uses a neural network developed through Reinforcement Learning to get a series of actions given a specific environment.

The project includes a series of scripts to generate the environment, train the agent, optimize the hyperparameters, and test the agent.
The project also include a Web API to get analysis results about the agent's performance.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

## Installation
Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/jonathanrgarza/ACScenarioAI

# Navigate to the project directory
cd ACScenarioAI

# Install dependencies
pip install -r requirements.txt

# Install the project (editable)
pip install -e .
```

## Usage
For running the project, you can use the following commands.

```bash
# Train the agent
ac_carrier_scenario agent train [options]

# Optimize the hyperparameters
ac_carrier_scenario agent optimize [options]

# Test the agent
ac_carrier_scenario agent enjoy [options]

# Run the Web API
ac_carrier_scenario api [options]

# Get help
ac_carrier_scenario --help
```

## Features
List of features included in the project.

- Optimize the hyperparameters of the agent
- Train an agent using Reinforcement Learning
- Test the agent
- Run a Web API to get analysis results for a given environment
- Process console arguments to run the project

## License
This project is for my personal purpose only.
You are allowed to clone the project and run it on your local machine for demonstration purposes only.
You are not allowed to use this project for any purposes without express permission.