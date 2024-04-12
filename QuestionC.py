#C for a

from dask.distributed import Client, progress
import time
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd
import seaborn as sns

n_processors = [10, 20]
n_processors_time = {}

for processor in n_processors:
    print(f"\n\n\nStarting computation with {processor} processors...\n\n\n")
    client = Client(n_workers=processor)

    # Question (a1)
    start_time = time.time()
    data1 = pd.read_csv("trips_by_distance.csv")
    data2 = pd.read_csv("trips_full_data.csv")
    # Calculate the average number of people staying at home per week
    data1['Population Staying at Home'] = data1['Population Staying at Home'].fillna(0)
    data1['Population Staying at Home'] = data1['Population Staying at Home'].round().astype('int64')
    average_per_week = data1.groupby('Week')['Population Staying at Home'].mean().round()

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.bar(average_per_week.index, average_per_week.values, color='purple', width=0.8)
    plt.title('Average Number of People Staying at Home per Week', fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Number of Weeks', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('Average Number of People', fontsize=14, fontweight='bold', color='black')
    plt.xticks(rotation=0, fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.show()
    end_time = time.time()
    
    # Question (a2)
    start_time = time.time()
    distance_ranges = [
        'Trips <1 Mile', 'Trips 1-3 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles', 'Trips 10-25 Miles',
        'Trips 1-25 Miles', 'Trips 25-50 Miles', 'Trips 50-100 Miles', 'Trips 25-100 Miles',
        'Trips 100-250 Miles', 'Trips 100+ Miles', 'Trips 250-500 Miles', 'Trips 500+ Miles'
    ]
    total_people_not_staying_home = data2['People Not Staying at Home'].mean()
    total_distances = {}
    for distance_range in distance_ranges:
        total_distance = (data2[distance_range] * data2['People Not Staying at Home']).mean()
        total_distances[distance_range] = total_distance

    plot_data = pd.DataFrame({
        'Distance Range (Miles)': distance_ranges,
        'Total Distance Traveled': [total_distances[distance_range] for distance_range in distance_ranges]
    })

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Distance Range (Miles)', y='Total Distance Traveled', data=plot_data, palette='viridis')
    plt.title('Total Distance Traveled for Each Type of Trip\n(Weighted by Number of People)', fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Distance Range (Miles)', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('Total Distance Traveled ', fontsize=14, fontweight='bold', color='black')
    plt.xticks(rotation=45, ha='right', fontsize=12, color='black')
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    end_time = time.time()
    
    dask_time = time.time() - start_time
    n_processors_time[processor] = dask_time
    print(f"\n\n\nTime Taken with {processor} processors: {dask_time} seconds\n\n\n")
    client.close()
    
# Print computation times
print("\n\n\n")
print("10 Processor:", n_processors_time[10], "seconds\n20 Processor:", n_processors_time[20], "seconds")
print("\n\n\n")












#C for b

import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from dask.distributed import Client, progress
import plotly.express as px
import time

# Define number of processors
n_processors = [10, 20]
n_processors_time = {}

# Read data using Pandas
data1 = pd.read_csv("Trips_By_Distance.csv")
# Define function to perform computation with different number of processors
def perform_computation(processor):
    print(f"\n\n\nStarting computation with {processor} processors...\n\n\n")
    client = Client(n_workers=processor)
    start = time.time()
    
    # Filter dates where more than 10,000,000 people conducted 10-25 trips
    data1_10_25 = data1[data1['Number of Trips 10-25'] > 10000000]

    # Filter dates where more than 10,000,000 people conducted 50-100 trips
    data1_50_100 = data1[data1['Number of Trips 50-100'] > 10000000]

    # Create scatter plot for 10-25 trips
    fig_10_25 = px.scatter(data1_10_25, x='Date', y='Number of Trips 10-25',
                           labels={'x': 'Date', 'y': 'Number of Trips 10-25'},
                           title='Dates with >10,000,000 people conducting 10-25 Trips')

    # Create scatter plot for 50-100 trips
    fig_50_100 = px.scatter(data1_50_100, x='Date', y='Number of Trips 50-100',
                            labels={'x': 'Date', 'y': 'Number of Trips 50-100'},
                            title='Dates with >10,000,000 people conducting 50-100 Trips')

    # Customizing appearance
    fig_10_25.update_traces(marker=dict(size=8, color='limegreen', line=dict(width=1, color='black')))
    fig_50_100.update_traces(marker=dict(size=8, color='coral', line=dict(width=1, color='black')))

    # Format x-axis
    fig_10_25.update_layout(xaxis=dict(tickformat="%Y-%m-%d", tickmode='auto', nticks=13))
    fig_50_100.update_layout(xaxis=dict(tickangle=0, tickformat="%Y-%m-%d", tickmode='auto', nticks=10))

    # Show the plots
    fig_10_25.show()
    fig_50_100.show()

    end = time.time()
    computation_time = end - start

    print(f"\n\n\nTime taken with {processor} processors: {computation_time} seconds\n\n\n")

    # Close the client after computation
    client.close()

    # Store computation time
    n_processors_time[processor] = computation_time


# Perform computation with different number of processors
for processor in n_processors:
    perform_computation(processor)

# Print computation times
print("\n\n\n")
print("10 Processor:", n_processors_time[10], "seconds\n20 Processor:", n_processors_time[20], "seconds")
print("\n\n\n")
