#!/bin/bash

# Must zip with 3.6 interpreter or else the competition will consider it invalid
$(python3.6 -m zipapp -p "/usr/bin/env python3.6" solution)

$(mv solution.pyz  evaluation/)
