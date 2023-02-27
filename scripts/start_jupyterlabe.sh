#!/usr/bin/env bash

poetry run jupyter-lab --ip 0.0.0.0 --no-browser --port 9029 --NotebookApp.token='' --NotebookApp.password='' &!
