#!/bin/bash

python -m venv venv
source venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
cd src
python data_preparation.py
python train.py
python predict.py
python evaluate.py
