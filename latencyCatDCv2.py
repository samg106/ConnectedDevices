import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from plotly.graph_objs import Sankey
import plotly.graph_objects as go
import re

# Function to categorize Value (ms) into specified groups
def categorize_value(ms):
    if ms < 5:
        return '<5'
    elif 5 <= ms <= 10:
        return '5-10'
    elif 10 <= ms <= 20:
        return '10-20'
    elif 20 <= ms <= 40:
        return '20-40'
    elif 40 <= ms <= 60:
        return '40-60'
    else:
        return '>60'

# Function to parse dates and ensure consistency
def parse_dates(date_str):
    for fmt in ('%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT

# Function to extract IP address without port number
def extract_ip(ip_with_port):
    match = re.match(r'(\d+\.\d+\.\d+\.\d+)', ip_with_port)
    return match.group(1) if match else ip_with_port

# Function to format IP addresses with a maximum of four per line
def format_ip_addresses(ips, per_line=4):
    lines = []
    for i in range(0, len(ips), per_line):
        lines.append(', '.join(ips[i:i + per_line]))
    return '<br>'.join(lines)

# Check for the correct number of command-line arguments
if len(sys.argv) != 3 or sys.argv[1] != '-f1':
    print("Usage: python script.py -f1 <filename>")
    sys.exit(1)

# Get the filename from the command line
filename = sys.argv[2]

# Check if the file exists
if not os.path.isfile(filename):
    print(f"File {filename} does not exist.")
    sys.exit(1)

# Read the CSV file
df = pd.read_csv(filename)

# Ensure the necessary columns are present
required_columns = ['Split (unit)', 'Value (ms)', 'Date', 'Split (datacenter)', 'Split (address)']
if not all(col in df.columns for col in required_columns):
    print(f"CSV file must contain the following columns: {', '.join(required_columns)}")
    sys.exit(1)

# Parse and ensure date consistency
df['Date'] = df['Date'].apply(parse_dates)

# Check for any NaT values resulting from conversion errors and handle them
if df['Date'].isna().any():
    print("There are some dates that couldn't be parsed. Ensure the date format is consistent.")
    sys.exit(1)

# Determine peak and off-peak times (customize as needed)
peak_hours = range(16, 23)
df['time_category'] = df['Date'].dt.hour.apply(lambda x: 'Peak' if x in peak_hours else 'Off-Peak')

# Categorize the Value (ms) into the specified groups
df['value_category'] = df['Value (ms)'].apply(categorize_value)

# Calculate counts for time categories and value categories
time_value_counts = df.groupby(['time_category', 'value_category']).size().reset_index(name='counts')

# Calculate counts for value categories and Split (datacenter)
value_datacenter_counts = df.groupby(['value_category', 'Split (datacenter)']).size().reset_index(name='counts')

# Calculate counts for Split (datacenter) and time categories
datacenter_time_counts = df.groupby(['Split (datacenter)', 'time_category']).size().reset_index(name='counts')

# Create Sankey diagram data
labels = ['Peak', 'Off-Peak', '<5', '5-10', '10-20', '20-40', '40-60', '>60']
label_to_index = {label: i for i, label in enumerate(labels)}

# Add unique Split (datacenter) to labels
unique_datacenters = df['Split (datacenter)'].unique()
datacenter_to_index = {datacenter: i + len(labels) for i, datacenter in enumerate(unique_datacenters)}
labels.extend(unique_datacenters)

# Add time category nodes again for the right side
right_time_categories = ['Peak', 'Off-Peak']
right_time_to_index = {label + ' (right)': i + len(labels) for i, label in enumerate(right_time_categories)}
labels.extend(right_time_categories)

# Calculate the total number of records
total_records = len(df)

# Calculate percentage of records for each node
node_percentages = [0] * len(labels)

# Populate node_percentages for time categories
for time_category in ['Peak', 'Off-Peak']:
    node_percentages[label_to_index[time_category]] = len(df[df['time_category'] == time_category]) / total_records * 100

# Populate node_percentages for value categories
for value_category in ['<5', '5-10', '10-20', '20-40', '40-60', '>60']:
    node_percentages[label_to_index[value_category]] = len(df[df['value_category'] == value_category]) / total_records * 100

# Populate node_percentages for datacenter nodes
for datacenter in unique_datacenters:
    node_percentages[datacenter_to_index[datacenter]] = len(df[df['Split (datacenter)'] == datacenter]) / total_records * 100

# Populate node_percentages for right-side time categories
for time_category in ['Peak', 'Off-Peak']:
    node_percentages[right_time_to_index[time_category + ' (right)']] = len(df[df['time_category'] == time_category]) / total_records * 100

source = []
target = []
value = []
link_colors = []
link_customdata = []

# Define colors for each value category node
value_node_colors = {
    '<5': 'rgba(0, 100, 0, 0.5)',        # dark green
    '5-10': 'rgba(144, 238, 144, 0.5)',  # light green
    '10-20': 'rgba(255, 165, 0, 0.5)',   # orange
    '20-40': 'rgba(255, 140, 0, 0.5)',   # dark orange
    '40-60': 'rgba(255, 0, 0, 0.5)',     # red
    '>60': 'rgba(128, 0, 0, 0.5)'        # dark red
}

# Define a different color for each datacenter node
datacenter_colors = [
    'rgba(0, 0, 255, 0.5)',     # blue
    'rgba(255, 0, 255, 0.5)',   # magenta
    'rgba(0, 255, 255, 0.5)',   # cyan
    'rgba(75, 0, 130, 0.5)',    # indigo
    'rgba(255, 215, 0, 0.5)'    # gold
]

# Create a list of node colors with placeholders
node_colors = [''] * len(labels)

# Set default colors for Peak and Off-Peak
node_colors[label_to_index['Peak']] = 'rgba(200, 200, 200, 0.8)'
node_colors[label_to_index['Off-Peak']] = 'rgba(200, 200, 200, 0.8)'

# Add colors for latency nodes
for value_category in ['<5', '5-10', '10-20', '20-40', '40-60', '>60']:
    node_colors[label_to_index[value_category]] = value_node_colors[value_category]

# Add colors for datacenter nodes, ensuring they do not clash with latency colors
for i, datacenter in enumerate(unique_datacenters):
    node_colors[datacenter_to_index[datacenter]] = datacenter_colors[i % len(datacenter_colors)]

# Add colors for the right-side time category nodes
node_colors[right_time_to_index['Peak (right)']] = 'rgba(200, 200, 200, 0.8)'
node_colors[right_time_to_index['Off-Peak (right)']] = 'rgba(200, 200, 200, 0.8)'

# Create customdata annotations for each node
node_customdata = [[] for _ in range(len(labels))]

# Populate customdata for latency nodes with unique IP addresses
for value_category in ['<5', '5-10', '10-20', '20-40', '40-60', '>60']:
    ips = df[df['value_category'] == value_category]['Split (address)'].apply(extract_ip).unique().tolist()
    formatted_ips = format_ip_addresses(list(set(ips)))  # Ensure unique IP addresses and format them
    percentage = node_percentages[label_to_index[value_category]]
    node_customdata[label_to_index[value_category]] = f"{formatted_ips}<br>Percentage: {percentage:.2f}%"

# Populate customdata for datacenter nodes with unique IP addresses
for datacenter in unique_datacenters:
    ips = df[df['Split (datacenter)'] == datacenter]['Split (address)'].apply(extract_ip).unique().tolist()
    formatted_ips = format_ip_addresses(list(set(ips)))  # Ensure unique IP addresses and format them
    percentage = node_percentages[datacenter_to_index[datacenter]]
    node_customdata[datacenter_to_index[datacenter]] = f"{formatted_ips}<br>Percentage: {percentage:.2f}%"

# Populate customdata for time category nodes with unique IP addresses
for time_category in ['Peak', 'Off-Peak']:
    ips = df[df['time_category'] == time_category]['Split (address)'].apply(extract_ip).unique().tolist()
    formatted_ips = format_ip_addresses(list(set(ips)))  # Ensure unique IP addresses and format them
    percentage = node_percentages[label_to_index[time_category]]
    node_customdata[label_to_index[time_category]] = f"{formatted_ips}<br>Percentage: {percentage:.2f}%"

# Populate customdata for right-side time category nodes with unique IP addresses
for time_category in ['Peak', 'Off-Peak']:
    ips = df[df['time_category'] == time_category]['Split (address)'].apply(extract_ip).unique().tolist()
    formatted_ips = format_ip_addresses(list(set(ips)))  # Ensure unique IP addresses and format them
    percentage = node_percentages[right_time_to_index[time_category + ' (right)']]
    node_customdata[right_time_to_index[time_category + ' (right)']] = f"{formatted_ips}<br>Percentage: {percentage:.2f}%"

# For time categories to value categories
for _, row in time_value_counts.iterrows():
    time_category = row['time_category']
    value_category = row['value_category']
    count = row['counts']
    target_total = df[df['value_category'] == value_category].shape[0]
    link_percentage = count / target_total * 100
    source.append(label_to_index[time_category])
    target.append(label_to_index[value_category])
    value.append(count)
    link_colors.append(value_node_colors[value_category])
    link_customdata.append(f"{link_percentage:.2f}%")

# For value categories to Split (datacenter)
for _, row in value_datacenter_counts.iterrows():
    value_category = row['value_category']
    datacenter = row['Split (datacenter)']
    count = row['counts']
    target_total = df[df['Split (datacenter)'] == datacenter].shape[0]
    link_percentage = count / target_total * 100
    source.append(label_to_index[value_category])
    target.append(datacenter_to_index[datacenter])
    value.append(count)
    link_colors.append(value_node_colors[value_category])
    link_customdata.append(f"{link_percentage:.2f}%")

# For Split (datacenter) to time categories (right side)
for _, row in datacenter_time_counts.iterrows():
    datacenter = row['Split (datacenter)']
    time_category = row['time_category']
    count = row['counts']
    target_total = df[df['time_category'] == time_category].shape[0]
    link_percentage = count / target_total * 100
    source.append(datacenter_to_index[datacenter])
    target.append(right_time_to_index[time_category + ' (right)'])
    value.append(count)
    link_colors.append('rgba(200, 200, 200, 0.5)')
    link_customdata.append(f"{link_percentage:.2f}%")

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=node_colors,
        customdata=node_customdata,
        hovertemplate='%{label}<br>%{customdata}<extra></extra>'
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors,
        customdata=link_customdata,
        hovertemplate='%{source.label} → %{target.label}<br>%{customdata}<extra></extra>'
    )
))



# Legend data
legend_labels = ['<5ms', '5-10ms', '10-20ms', '20-40ms', '40-60ms']
legend_colors = ['rgba(0, 100, 0, 0.7)', 'rgba(144, 238, 144, 0.7)', 'rgba(255, 165, 0, 0.7)', 'rgba(255, 140, 0, 0.7)', 'rgba(255, 0, 0, 0.7)']

# Calculate x positions for the legend annotations to be equally spaced
legend_x_positions = np.linspace(0.1, 0.9, len(legend_labels))

# Create legend annotations with calculated x positions and borders
annotations = [
    dict(
        x=x, y=-0.1, xref='paper', yref='paper',
        text=f'<span style="color:{color};">{label}</span>',
        showarrow=False,
        font=dict(size=12),
        bordercolor=color,
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        opacity=0.8
    )
    for x, label, color in zip(legend_x_positions, legend_labels, legend_colors)
]

# Calculate dynamic metrics
total_agents = df['Split (unit)'].nunique()
total_datacenters = df['Split (datacenter)'].nunique()
lowest_value_categories = df[df['value_category'].isin(['<5', '5-10'])]
optimal_location = lowest_value_categories['Split (datacenter)'].value_counts().idxmax()

# Total duration of dataset (days)
total_duration = (df['Date'].max() - df['Date'].min()).days

# Optimal location total usage
optimal_location_total_usage = len(lowest_value_categories[lowest_value_categories['Split (datacenter)'] == optimal_location]) / total_records * 100

# Create dynamic metrics text
dynamic_metrics_text_grp1 = (
    f"Total amount of agents: {total_agents}, "
    f"Total amount of datacenters: {total_datacenters}, "
    f"Total duration of dataset: {total_duration} days, "
)
# Create dynamic metrics text
dynamic_metrics_text_grp2 = (
    f"Most optimal/preferred location: {optimal_location}, "
    f"Optimal location total usage: {optimal_location_total_usage:.2f}%"
)





# Update layout with legend annotations and title
fig.update_layout(
    title={
        'text': (
            "eGaming - Investigating Content Location: Latency (ms) and Peak/Off-Peak Relationship<br>"
            f"<br>"
            f"<span style='font-size:14px;'>{dynamic_metrics_text_grp1}</span>"
             f"<br>"
            f"<span style='font-size:14px;'>{dynamic_metrics_text_grp2}</span>"
        ),
      
    },
    font_size=10,
    annotations=annotations
)
fig.show()
