#!/bin/bash

$(python -m zipapp -p "/usr/bin/env python" solution)

$(mv solution.pyz  evaluation/)
