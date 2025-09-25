#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Run the Streamlit UI
poetry run streamlit run scripts/ui/ui_v3.py