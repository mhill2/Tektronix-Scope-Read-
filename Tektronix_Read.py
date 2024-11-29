import pandas as pd
import matplotlib.pyplot as plt
import plthelper
import itertools

plthelper.init()

# User option to separate channels into different plots
separate_plots = False  # Set to False for single plot, True for separate plots

# Determine the number of rows to skip until "TIME" or "CH" columns appear
with open('data.csv', 'r') as file: # change data.csv if file has a different neam
    for i, line in enumerate(file):
        if 'TIME' in line and 'CH' in line:  # Adjust to check for both 'TIME' and 'CH'
            skiprows = i
            break

# Load the CSV file into a DataFrame, skipping determined rows
df = pd.read_csv('data.csv', skiprows=skiprows)

# Rename columns to remove leading/trailing spaces
df.columns = df.columns.str.strip()

# Identify the TIME column name (in case it's not exactly 'TIME')
time_column = next((col for col in df.columns if 'TIME' in col.upper()), None)
if not time_column:
    raise KeyError("The TIME column could not be found in the CSV file.")

# Identify columns that start with "CH"
channel_columns = [col for col in df.columns if col.startswith("CH")]

# Use a color cycle for separate plots
color_cycle = plt.cm.tab10.colors  # Or you can use plt.cm.viridis.colors or any other colormap
color_iterator = itertools.cycle(color_cycle)

if separate_plots:
    # Plot each channel in a separate figure with a different color
    for ch in channel_columns:
        color = next(color_iterator)  # Get the next color from the cycle
        plt.figure(figsize=(10, 6))
        plt.plot(df[time_column], df[ch], label=f'{ch} (Voltage)', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Time vs Voltage for {ch}')
        plt.legend()
        plt.grid(True)
        plt.show()  # Show each plot individually
else:
    # Plot all channels in a single figure
    plt.figure(figsize=(10, 6))
    for ch in channel_columns:
        plt.plot(df[time_column], df[ch], label=f'{ch} (Voltage)')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Time vs Voltage for All Channels')
    plt.legend()
    plt.grid(True)
    plt.show()