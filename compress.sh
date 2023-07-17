#!/bin/bash

$(python3.6 -m zipapp -p "/usr/bin/env python3.6" solution)

$(mv solution.pyz  evaluation/)
