import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Step 1: Read the data from the CSV file using Pandas
data = pd.read_csv('output/agent_pos_gut.csv')

# Step 2: Prepare the data
ticks = sorted(data['tick'].unique())  # Get unique ticks


# Step 3: Create the animation function
def update(frame):
    plt.cla()  # Clear the current plot
    plt.xlim(0, 200)  # Set X-axis limit
    plt.ylim(0, 200)  # Set Y-axis limit
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bifido in red Agent Movement over Time (Tick {})'.format(frame))

    # Draw grid
    plt.grid(True)

    # Filter data for the current frame
    frame_data = data[data['tick'] == frame]

    # Plot points for agent with ID 2 in blue and agent with ID 4 in red
    agent_2_data = frame_data[frame_data['agent_id'] == 2]
    plt.scatter(agent_2_data['x'], agent_2_data['y'], color='blue')

    agent_4_data = frame_data[frame_data['agent_id'] == 4]
    plt.scatter(agent_4_data['x'], agent_4_data['y'], color='red')

    agent_3_data = frame_data[frame_data['agent_id'] == 3]
    plt.scatter(agent_3_data['x'], agent_3_data['y'], color='yellow')

    agent_5_data = frame_data[frame_data['agent_id'] == 5]
    plt.scatter(agent_5_data['x'], agent_5_data['y'], color='green')

    agent_1_data = frame_data[frame_data['agent_id'] == 1]
    plt.scatter(agent_1_data['x'], agent_1_data['y'], color='black')



# Step 4: Create the animation
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

# Connect the key press event to the function
fig.canvas.mpl_connect('key_press_event', toggle_animation)

# Prevent closing the program when 'q' key is pressed
plt.gcf().canvas.mpl_disconnect(plt.gcf().canvas.manager.key_press_handler_id)
# Show the plot
plt.show()