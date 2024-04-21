#!/bin/bash

# Navigate to the project root if your current directory is the script's directory
# cd "${0%/*}/.."

# Set the FLASK_APP environment variable
export FLASK_APP=src/server/server.py

# Enable Flask development mode to activate debugger and reloader
# export FLASK_ENV=production
# export FLASK_ENV=development
# export FLASK_DEBUG=true

# Start the Flask server
flask run --host=0.0.0.0

