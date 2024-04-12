#B

import pandas as pd
import plotly.express as px
import time
data1 = pd.read_csv("trips_by_distance.csv")


# Record start time
start_time = time.time()

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

# Record end time
end_time = time.time()

# Calculate and print the time taken
print("Time taken for serial processing to execute the code:", end_time - start_time, "seconds")