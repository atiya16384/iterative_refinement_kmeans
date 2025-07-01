#!/bin/bash
rm -rf .venv
$(which python3) -m venv .venv
source .venv/bin/activate 
pip install --upgrade pip
python3 -m pip install numpy pandas matplotlib scikit-learn 
pip install git+https://github.com/IntelPython/scikit-learn_bench.git
python3 -m pip install scikit-learn-bench