num_sats = 3
method = "TDoA+FDoA"
std_bearing_deg = 1
std_tdoa = 100e-9
std_fdoa = 10
num_simulations = 500
lat_e = 19.152120
long_e = 72.900804
num_sample_points = 1
weights = {"AoA" : 1, "TDoA" : 1e12, "FDoA" : 1e2}
tag = f"TRL4_{method}_ns{num_sats}_nsim{num_simulations}_f{std_fdoa}_nsam{num_sample_points}"
