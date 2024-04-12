#E

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Trips_Full_Data.csv")

# Aggregate the data by summing up the number of travelers for each distance range
distance_columns = [
   'Trips <1 Mile', 'Trips 1-3 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles', 'Trips 10-25 Miles', 
                   'Trips 25-50 Miles', 'Trips 50-100 Miles', 'Trips 100-250 Miles', 'Trips 250-500 Miles', 
                   'Trips 500+ Miles'
                    ]
total_travelers_by_distance = data[distance_columns].sum()

# Plotting
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
colors = sns.color_palette("viridis", len(total_travelers_by_distance))
total_travelers_by_distance.plot(kind='bar', color=colors)
plt.title('Number of Participants by Distance-Trips', fontsize=16, fontweight='bold', color='black')
plt.xlabel('Distance Category', fontsize=14, fontweight='bold', color='black')
plt.ylabel('Number of Trips', fontsize=14, fontweight='bold', color='black')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
plt.show()
