#!/bin/bash
# rm combined_data.csv
for n in {1..200};
do
   python3 3d_doppler_search.py
   python3 plot_cluster.py
done