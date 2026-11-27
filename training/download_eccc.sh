#!/bin/sh
# GenAI declaration: written with Claude (Anthropic) for the SEH500 project.
# Download ECCC hourly data, one CSV per station per month, into ./eccc
# 51459 = TORONTO INTL A (Pearson), 31688 = TORONTO CITY (downtown)
mkdir -p eccc
for id in 31688 51459; do
  for y in $(seq 2014 2025); do
    for m in $(seq 1 12); do echo "$id $y $m"; done
  done
done | xargs -P 8 -n 3 sh -c 'curl -s --retry 3 -o "eccc/h_$0_$1_$(printf %02d $2).csv" "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=$0&Year=$1&Month=$2&Day=1&timeframe=1&submit=Download+Data"'
