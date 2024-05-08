import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Step 1: Read the data from the CSV file using Pandas
data = pd.read_csv('output/agent_pos_cns.csv')

# Step 2: Prepare the data
ticks = sorted(data['tick'].unique())  # Get unique ticks

# Step 3: Create a dictionary to map neuron states to colors
state_colors = {
    0: 'green',
    1: 'orange',
    2: 'red',
    3: 'black'
}

# Step 4: Create the animation function
def update(frame):
    plt.cla()  # Clear the current plot
    plt.xlim(0, 200)  # Set X-axis limit
    plt.ylim(0, 200)  # Set Y-axis limit
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agent Movement over Time (Tick {})'.format(frame))

    # Draw grid
    plt.grid(True)

    # Filter data for the current frame
    frame_data = data[data['tick'] == frame]

    # Plot points for agents
    agent_1_data = frame_data[frame_data['agent_id'] == 1]
    agent_1_scatter = plt.scatter(agent_1_data['x'], agent_1_data['y'], color='blue')

    other_agents_data = frame_data[frame_data['agent_id'] != 1]
    other_agents_scatter = plt.scatter(other_agents_data['x'], other_agents_data['y'],
                                        c=other_agents_data['neuron_state'].map(state_colors).fillna('blue'))

    return [agent_1_scatter, other_agents_scatter]

# Step 5: Create the animation
fig = plt.figure(figsize=(8, 6))
ani = FuncAnimation(fig, update, frames=ticks, interval=50, repeat=False)

# Variable to track animation status
animation_running = True

# Function to toggle animation when 'q' key is pressed
def toggle_animation(event):
    global animation_running
    if event.key == 'q':
        if animation_running:
            ani.event_source.stop()  # Pause animation
        else:
            ani.event_source.start()  # Resume animation
        animation_running = not animation_running
    elif event.key == 'r':  # Restart animation when 'r' key is pressed
        ani.frame_seq = ani.new_frame_seq()
        ani.event_source.start()
        animation_running = True

# Connect the key press event to the function
fig.canvas.mpl_connect('key_press_event', toggle_animation)

# Prevent closing the program when 'q' key is pressed
plt.gcf().canvas.mpl_disconnect(plt.gcf().canvas.manager.key_press_handler_id)

# Show the plot
plt.show()
