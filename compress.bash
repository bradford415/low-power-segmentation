#!/bin/bash

$(python -m zipapp -p "/usr/bin/env python3" compression)

$(cp compression.pyz  evaluation/)
