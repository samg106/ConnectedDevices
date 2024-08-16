import pandas as pd
import numpy as np
import sys
import os
import requests
import plotly.graph_objects as go

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
    else:
        return '40-60'

# Function to parse dates and ensure consistency
def parse_dates(date_str):
    for fmt in ('%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT

# Function to enrich the CSV file with IPinfo data
def enrich_csv_with_ipinfo(df, api_key):
    unique_ips = df['Split (address)'].unique()
    dfIPinfoResults = pd.DataFrame(unique_ips, columns=['Split (address)'])
    columns = ['ip', 'hostname', 'city', 'region', 'country', 'loc', 'org', 'postal', 'timezone', 'asn']
    for column in columns:
        dfIPinfoResults[column] = None
    
    for index, row in dfIPinfoResults.iterrows():
        ip = row['Split (address)']
        try:
            response = requests.get(f"https://ipinfo.io/{ip}/json?token={api_key}")
            results = response.json()
            for column in columns:
                if column == 'asn' and 'asn' in results:
                    asn_info = results['asn']
                    dfIPinfoResults.at[index, column] = asn_info['asn']
                else:
                    dfIPinfoResults.at[index, column] = results.get(column, 'Unknown')
        except Exception as e:
            print(f"Error retrieving data for {ip}: {e}")
            for column in columns:
                dfIPinfoResults.at[index, column] = 'Lookup failed'
    
    df = pd.merge(df, dfIPinfoResults, on='Split (address)', how='left')
    return df

# Check for the correct number of command-line arguments
if len(sys.argv) < 3 or sys.argv[1] != '-f1':
    print("Usage: python script.py -f1 <filename> [-u] [-meta <column_name>] [-api <api_key>]")
    sys.exit(1)

# Get the filename from the command line
filename = sys.argv[2]

# Check if the '-u' parameter is used
use_unit_time_category = '-u' in sys.argv

# Check if the '-meta' parameter is used and get the column name
meta_column = None
if '-meta' in sys.argv:
    meta_index = sys.argv.index('-meta')
    if meta_index + 1 < len(sys.argv):
        meta_column = sys.argv[meta_index + 1]
    else:
        print("Usage: python script.py -f1 <filename> [-u] [-meta <column_name>] [-api <api_key>]")
        sys.exit(1)

# Check if the '-api' parameter is used and get the API key
api_key = None
if '-api' in sys.argv:
    api_index = sys.argv.index('-api')
    if api_index + 1 < len(sys.argv):
        api_key = sys.argv[api_index + 1]
    else:
        print("Usage: python script.py -f1 <filename> [-u] [-meta <column_name>] [-api <api_key>]")
        sys.exit(1)

# Check if the file exists
if not os.path.isfile(filename):
    print(f"File {filename} does not exist.")
    sys.exit(1)

# Read the CSV file
df = pd.read_csv(filename)

# Ensure the necessary columns are present
required_columns = ['Date', 'Split (media)', 'Split (unit)', 'Split (address)', 'Value (ms)', 'hostname', 'city', 'region', 'country', 'org']
if not all(col in df.columns for col in required_columns):
    print(f"CSV file must contain the following columns: {', '.join(required_columns)}")
    sys.exit(1)

# Check if the CSV has been enriched by verifying the presence of enrichment columns
enrichment_columns = ['ip', 'hostname', 'city', 'region', 'country', 'loc', 'org', 'postal', 'timezone', 'asn']
if not all(col in df.columns for col in enrichment_columns):
    if api_key:
        print("CSV file is not enriched. Enriching the CSV file with IPinfo data...")
        df = enrich_csv_with_ipinfo(df, api_key)
        enriched_filename = 'enriched_' + os.path.basename(filename)
        df.to_csv(enriched_filename, index=False)
        print(f"CSV file has been enriched and saved as {enriched_filename}.")
    else:
        print("CSV file is not enriched and no API key provided. Enrichment skipped.")
else:
    print("CSV file is already enriched.")

# Parse and ensure date consistency
df['Date'] = df['Date'].apply(parse_dates)

# Check for any NaT values resulting from conversion errors and handle them
if df['Date'].isna().any():
    print("There are some dates that couldn't be parsed. Ensure the date format is consistent.")
    sys.exit(1)

# Determine the column to use for unit/category
unit_column = meta_column if meta_column else 'Split (unit)'

# Ensure the selected unit column is treated as a string
df[unit_column] = df[unit_column].astype(str)

# Create a combined Unit ID with time category if '-u' parameter is used
if use_unit_time_category:
    # Determine peak and off-peak times (customize as needed)
    peak_hours = range(16, 23)
    df['time_category'] = df['Date'].dt.hour.apply(lambda x: 'Peak' if x in peak_hours else 'Off-Peak')
    df['unit_time_category'] = df[unit_column] + ' (' + df['time_category'] + ')'

# Categorize the Value (ms) into the specified groups
df['value_category'] = df['Value (ms)'].apply(categorize_value)

# Group by the specified columns and count unique unit ids
if use_unit_time_category:
    grouped = df.groupby(['Split (media)', 'unit_time_category', 'value_category', 'city', 'region', 'country', 'org'])[unit_column].nunique().reset_index(name='counts')
else:
    grouped = df.groupby(['Split (media)', unit_column, 'value_category', 'city', 'region', 'country', 'org'])[unit_column].nunique().reset_index(name='counts')

# Identify and modify flows to orgs that include 'AS63293' or 'AS32934'
special_asns = ['AS63293', 'AS32934']
grouped['special_asn'] = grouped['org'].apply(lambda x: any(asn in x for asn in special_asns))

# Create Sankey diagram data
labels = []
node_customdata = []

# Add unique values for each node category
if use_unit_time_category:
    for col in ['Split (media)', 'unit_time_category', 'city', 'region', 'country', 'org']:
        unique_values = grouped[col].unique()
        labels.extend(unique_values)
elif meta_column:
    for col in ['Split (media)', unit_column, 'city', 'region', 'country', 'org']:
        unique_values = grouped[col].unique()
        labels.extend(unique_values)
else:
    for col in ['Split (media)', 'city', 'region', 'country', 'org']:
        unique_values = grouped[col].unique()
        labels.extend(unique_values)

# Include special nodes for AS63293 and AS32934
if use_unit_time_category:
    for special_asn in special_asns:
        for col in ['city', 'region', 'country', 'org']:
            labels.extend([f"{val} ({special_asn})" for val in grouped.loc[grouped['special_asn'], col].unique()])

elif meta_column:
    for special_asn in special_asns:
        for col in ['city', 'region', 'country', 'org']:
            labels.extend([f"{val} ({special_asn})" for val in grouped.loc[grouped['special_asn'], col].unique()])
else:
    for special_asn in special_asns:
        for col in ['city', 'region', 'country', 'org']:
            labels.extend([f"{val} ({special_asn})" for val in grouped.loc[grouped['special_asn'], col].unique()])

# Remove duplicates and create a mapping
labels = list(dict.fromkeys(labels))
label_to_index = {label: i for i, label in enumerate(labels)}

source = []
target = []
value = []
link_color = []
link_customdata = []

# Define colors for different value categories
value_colors = {
    '<5': 'rgba(0, 100, 0, 0.5)',        # dark green
    '5-10': 'rgba(144, 238, 144, 0.5)',  # light green
    '10-20': 'rgba(255, 165, 0, 0.5)',   # orange
    '20-40': 'rgba(255, 140, 0, 0.5)',   # dark orange
    '40-60': 'rgba(255, 0, 0, 0.5)'      # red
}

# Add links for the Sankey diagram
flow_dict = {}

def add_flow(src, tgt, val, color, customdata=None):
    key = (src, tgt)
    if key in flow_dict:
        flow_dict[key][0] += val
    else:
        flow_dict[key] = [val, color, customdata if customdata else val]

if use_unit_time_category:
    for _, row in grouped.iterrows():
        if row['special_asn']:
            for special_asn in special_asns:
                if special_asn in row['org']:
                    # Add links with special ASN handling
                    add_flow(row['Split (media)'], row['unit_time_category'], row['counts'], value_colors[row['value_category']])
                    add_flow(row['unit_time_category'], row['city'] + f' ({special_asn})', row['counts'], value_colors[row['value_category']])
                    if row['city'] == row['region']:
                        add_flow(row['city'] + f' ({special_asn})', row['country'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                    else:
                        add_flow(row['city'] + f' ({special_asn})', row['region'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                        add_flow(row['region'] + f' ({special_asn})', row['country'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                    add_flow(row['country'] + f' ({special_asn})', row['org'], row['counts'], 'rgba(128, 128, 128, 0.5)')
        else:
            add_flow(row['Split (media)'], row['unit_time_category'], row['counts'], value_colors[row['value_category']])
            add_flow(row['unit_time_category'], row['city'], row['counts'], value_colors[row['value_category']])
            if row['city'] == row['region']:
                add_flow(row['city'], row['country'], row['counts'], 'rgba(128, 128, 128, 0.5)')
            else:
                add_flow(row['city'], row['region'], row['counts'], 'rgba(128, 128, 128, 0.5)')
                add_flow(row['region'], row['country'], row['counts'], 'rgba(128, 128, 128, 0.5)')
            add_flow(row['country'], row['org'], row['counts'], 'rgba(128, 128, 128, 0.5)')

elif meta_column:
    for _, row in grouped.iterrows():
        if row['special_asn']:
            for special_asn in special_asns:
                if special_asn in row['org']:
                    # Add links with special ASN handling
                    add_flow(row['Split (media)'], row[unit_column], row['counts'], value_colors[row['value_category']])
                    add_flow(row[unit_column], row['city'] + f' ({special_asn})', row['counts'], value_colors[row['value_category']])
                    if row['city'] == row['region']:
                        add_flow(row['city'] + f' ({special_asn})', row['country'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                    else:
                        add_flow(row['city'] + f' ({special_asn})', row['region'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                        add_flow(row['region'] + f' ({special_asn})', row['country'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                    add_flow(row['country'] + f' ({special_asn})', row['org'], row['counts'], 'rgba(128, 128, 128, 0.5)')
        else:
            add_flow(row['Split (media)'], row[unit_column], row['counts'], value_colors[row['value_category']])
            add_flow(row[unit_column], row['city'], row['counts'], value_colors[row['value_category']])
            if row['city'] == row['region']:
                add_flow(row['city'], row['country'], row['counts'], 'rgba(128, 128, 128, 0.5)')
            else:
                add_flow(row['city'], row['region'], row['counts'], 'rgba(128, 128, 128, 0.5)')
                add_flow(row['region'], row['country'], row['counts'], 'rgba(128, 128, 128, 0.5)')
            add_flow(row['country'], row['org'], row['counts'], 'rgba(128, 128, 128, 0.5)')

else:
    for _, row in grouped.iterrows():
        if row['special_asn']:
            for special_asn in special_asns:
                if special_asn in row['org']:
                    # Add links with special ASN handling
                    add_flow(row['Split (media)'], row['city'] + f' ({special_asn})', row['counts'], value_colors[row['value_category']])
                    if row['city'] == row['region']:
                        add_flow(row['city'] + f' ({special_asn})', row['country'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                    else:
                        add_flow(row['city'] + f' ({special_asn})', row['region'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                        add_flow(row['region'] + f' ({special_asn})', row['country'] + f' ({special_asn})', row['counts'], 'rgba(128, 128, 128, 0.5)')
                    add_flow(row['country'] + f' ({special_asn})', row['org'], row['counts'], 'rgba(128, 128, 128, 0.5)')
        else:
            add_flow(row['Split (media)'], row['city'], row['counts'], value_colors[row['value_category']])
            if row['city'] == row['region']:
                add_flow(row['city'], row['country'], row['counts'], 'rgba(128, 128, 128, 0.5)')
            else:
                add_flow(row['city'], row['region'], row['counts'], 'rgba(128, 128, 128, 0.5)')
                add_flow(row['region'], row['country'], row['counts'], 'rgba(128, 128, 128, 0.5)')
            add_flow(row['country'], row['org'], row['counts'], 'rgba(128, 128, 128, 0.5)')

# Create source, target, value lists from the combined flow dictionary
for (src, tgt), (val, color, customdata) in flow_dict.items():
    source.append(label_to_index[src])
    target.append(label_to_index[tgt])
    value.append(val)
    link_color.append(color)
    link_customdata.append(customdata)

# Calculate the percentage for each link
target_node_counts = {label: 0 for label in labels}
source_node_counts = {label: 0 for label in labels}
for src, tgt, val in zip(source, target, value):
    target_node_counts[labels[tgt]] += val
    source_node_counts[labels[src]] += val

link_percentages = [(val / target_node_counts[labels[tgt]]) * 100 if target_node_counts[labels[tgt]] > 0 else 0 for tgt, val in zip(target, value)]
source_percentages = [(val / source_node_counts[labels[src]]) * 100 if source_node_counts[labels[src]] > 0 else 0 for src, val in zip(source, value)]

# Create customdata for city nodes with IP addresses and media type percentages
def format_ip_addresses(addresses):
    lines = []
    for i in range(0, len(addresses), 4):
        lines.append(", ".join(addresses[i:i + 4]))
    return "<br>".join(lines)

for label in labels:
    if label in df['city'].unique():
        addresses = df[df['city'] == label]['Split (address)'].unique()
        city_data = df[df['city'] == label]
    elif any(label.startswith(city) for city in df['city'].unique()):
        base_label = " ".join(label.split(" ")[:-1])  # Remove the ASN part
        addresses = df[df['city'] == base_label]['Split (address)'].unique()
        city_data = df[df['city'] == base_label]
    else:
        addresses = []
        city_data = pd.DataFrame()

    if not city_data.empty:
        media_counts = city_data['Split (media)'].value_counts(normalize=True) * 100
        media_percentage_str = "<br>".join([f"{media}: {perc:.2f}%" for media, perc in media_counts.items()])
    else:
        media_percentage_str = ""

    customdata_entry = format_ip_addresses(addresses) + "<br>" + media_percentage_str
    node_customdata.append(customdata_entry)

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        customdata=node_customdata,
        hovertemplate='%{label}<br>%{customdata}<extra></extra>'
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_color,
        customdata=[f"Source Percentage: {sp:.2f}%, Target Percentage: {tp:.2f}%" for sp, tp in zip(source_percentages, link_percentages)],
        hovertemplate='%{source.label} -> %{target.label}<br>%{customdata}<extra></extra>'
    )
))

# Update title based on the diagram type
if use_unit_time_category:
    title_text = "Facebook Messenger CDN Analysis: Media Type > Unit ID (Peak/Off-Peak) > City > Region > Country > Org"
elif meta_column:
    title_text = f"Facebook Messenger CDN Analysis: Media Type > {unit_column.capitalize()} > City > Region > Country > Org"
else:
    title_text = "Facebook Messenger CDN Analysis: Media Type > City > Region > Country > Org"

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

# Update layout with legend annotations
fig.update_layout(
    title_text=title_text,
    font_size=10,
    annotations=annotations
)

fig.show()
