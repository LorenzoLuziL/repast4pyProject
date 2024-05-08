
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Step 1: Read the data from the file using Pandas
data = pd.read_csv('output/agent_counts.csv')

# Step 2: Extract data for plotting
ticks = data['tick']
alpha_counts = data['alpha_count']
bifido_counts = data['bifido_count']
alpha_gut_counts = data['alpha_gut_count']
LPS_counts = data['LPS_count']
gram_negative_counts = data['gram_negative_count']

def update(frame):
    plt.cla()  # Clear the current plot
    plt.plot(ticks[:frame], alpha_counts[:frame], label='Alpha Count')
    plt.plot(ticks[:frame], bifido_counts[:frame], label='Bifido Count')
    plt.plot(ticks[:frame], alpha_gut_counts[:frame], label='Alpha Gut Count')
    plt.plot(ticks[:frame], LPS_counts[:frame], label='LPS Count')
    plt.plot(ticks[:frame], gram_negative_counts[:frame], label='Gram Negative Count')

    plt.xlabel('Tick')
    plt.ylabel('Count')
    plt.title('Counts over Time')
    plt.legend()
    plt.grid(True)

# Step 4: Create the animation
fig = plt.figure(figsize=(10, 6))
ani = FuncAnimation(fig, update, frames=len(ticks), interval=50, repeat=False)



plt.show()
