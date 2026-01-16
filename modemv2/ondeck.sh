#!/bin/bash

for i in {1..5}; do
    echo "Run $i"
    sh scripts/franka/bin_pick/modemv2.sh
    sleep 20
done
