#!/usr/bin/env bash

poetry run jupyter-lab --ip 0.0.0.0 --no-browser --port 9999 --NotebookApp.token='' --NotebookApp.password='' &!
