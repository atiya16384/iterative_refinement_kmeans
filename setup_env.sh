#!/bin/bash
rm -rf .venv
$(which python3) -m venv .venv
source .venv/bin/activate 
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn 