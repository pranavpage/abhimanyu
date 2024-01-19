import matplotlib.pyplot as plt
import pandas as pd

# Specify the path to your CSV file
csv_file_path = './data/accuracy.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Group the data by 'Number of sats'
grouped = df.groupby('Number of sats')
# Create a figure and axis
fig, ax = plt.subplots()

# Loop through the grouped data and plot each curve
for num_sats, group_data in grouped:
    x_values = group_data['Number of measurements']
    y_values = group_data['area']
    ax.plot(x_values, y_values, marker='o', label=f'{num_sats} sats')
ax.set_xticks([1, 5, 10, 25])

# ax.set_xscale('log')
# ax.set_yscale('log')

# Set labels and title
ax.set_xlabel('Number of Measurements')
ax.set_ylabel('Area of Localization (sq km)')
ax.set_title('Geolocation accuracy for combined TDoA+FDoA')

# Add a legend
ax.legend()

# Show the plot
plt.grid(True, which="both")
plt.savefig("./plots/accuracy_v2.png")
plt.show()