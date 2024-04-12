#A

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

# Record start time
start_time = time.time()

data1 = pd.read_csv("trips_by_distance.csv")
data2 = pd.read_csv("trips_full_data.csv")

#A1
# Calculate the average number of people staying at home per week
# Fill null values with 0/NaN (depending on what you want)
data1['Population Staying at Home'] = data1['Population Staying at Home'].fillna(0)
# Round floats to ints
data1['Population Staying at Home'] = data1['Population Staying at Home'].round().astype('int64')
average_per_week = data1.groupby('Week')['Population Staying at Home'].mean().round()

# Plotting the data
plt.figure(figsize=(10, 6))
plt.bar(average_per_week.index, average_per_week.values, color='purple', width=0.8)
plt.title('Average Number of People Staying at Home per Week', fontsize=16, fontweight='bold', color='black')
plt.xlabel('Number of Weeks', fontsize=14, fontweight='bold', color='black')
plt.ylabel('Average Number of People', fontsize=14, fontweight='bold', color='black')
plt.xticks(rotation=0, fontsize=12, color='black')  
plt.yticks(fontsize=12, color='black')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
plt.tight_layout()
plt.show()


#A2

# Define the order of distance ranges
distance_ranges = [
     'Trips <1 Mile', 'Trips 1-3 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles', 'Trips 10-25 Miles', 
                   'Trips 25-50 Miles', 'Trips 50-100 Miles', 'Trips 100-250 Miles', 
                   'Trips 500+ Miles'
]
# Calculate the total number of people not staying at home per week
total_people_not_staying_home = data2['People Not Staying at Home'].mean()
# Calculate the total distance traveled for each type of trip
total_distances = {}
for distance_range in distance_ranges:
    total_distance = (data2[distance_range] * data2['People Not Staying at Home']).mean()
    total_distances[distance_range] = total_distance
# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Distance Range (Miles)': distance_ranges,
    'Total Distance Traveled': [total_distances[distance_range] for distance_range in distance_ranges]
})

# Plotting with Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='Distance Range (Miles)', y='Total Distance Traveled', data=plot_data, palette='viridis')
plt.title('Total Distance Traveled for Each Type of Trip\n(Weighted by Number of People)', fontsize=16, fontweight='bold', color='black')
plt.xlabel('Distance Range (Miles)', fontsize=14, fontweight='bold', color='black')
plt.ylabel('Total Distance Traveled ', fontsize=14, fontweight='bold', color='black')
plt.xticks(rotation=45, ha='right',fontsize=12, color='black')  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Record end time
end_time = time.time()
# Calculate and print the time taken
print("Time taken for serial processing to execute the code:", end_time - start_time, "seconds")