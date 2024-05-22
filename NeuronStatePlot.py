
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Step 1: Read the data from the file using Pandas
data = pd.read_csv('output/neuron_state.csv')

# Step 2: Extract data for plotting
ticks = data['tick']
balanced_state = data['balanced']
misfolding_state = data['misfolding']
oligomer_state = data['oligomer']
lewy_bodies = data['lewy_bodies']


def update(frame):
    plt.cla()  # Clear the current plot
    plt.plot(ticks[:frame], balanced_state[:frame], label='Balanced State')
    plt.plot(ticks[:frame], misfolding_state[:frame], label='Misfolding State')
    plt.plot(ticks[:frame], oligomer_state[:frame], label='Oligomer State')
    plt.plot(ticks[:frame], lewy_bodies[:frame], label='Lewy boides State')

    plt.xlabel('Tick')
    plt.ylabel('Count')
    plt.title('Counts over Time')
    plt.legend()
    plt.grid(True)

# Step 4: Create the animation
fig = plt.figure(figsize=(10, 6))
ani = FuncAnimation(fig, update, frames=len(ticks), interval=50, repeat=False)



plt.show()
