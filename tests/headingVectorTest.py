import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Read CSV file
csv_file = 'vectors.csv'  # Change this to your CSV filename
data = pd.read_csv(csv_file)

# Assuming CSV columns are: v1_x, v1_y, v2_x, v2_y
# Adjust column names if needed
v1_x = data.iloc[:, 0].values
v1_y = data.iloc[:, 1].values
v2_x = data.iloc[:, 2].values
v2_y = data.iloc[:, 3].values

dt = 0.1  # Time step in seconds
time = np.arange(len(v1_x)) * dt

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Determine axis limits based on data
max_val = max(np.max(np.abs(v1_x)), np.max(np.abs(v1_y)),
              np.max(np.abs(v2_x)), np.max(np.abs(v2_y)))
margin = max_val * 0.1
ax.set_xlim(-max_val - margin, max_val + margin)
ax.set_ylim(-max_val - margin, max_val + margin)

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Animated 2D Vectors', fontsize=14)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_aspect('equal')

# Initialize vector plots (quiver)
quiver1 = ax.quiver(0, 0, v1_x[0], v1_y[0], angles='xy', scale_units='xy', 
                    scale=1, color='blue', width=0.006, label='Heading')
quiver2 = ax.quiver(0, 0, v2_x[0], v2_y[0], angles='xy', scale_units='xy', 
                    scale=1, color='red', width=0.006, label='Target Heading')

# Time text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.legend(loc='upper right')

def init():
    """Initialize animation"""
    quiver1.set_UVC(v1_x[0], v1_y[0])
    quiver2.set_UVC(v2_x[0], v2_y[0])
    time_text.set_text('')
    return quiver1, quiver2, time_text

def animate(frame):
    """Update function for animation"""
    # Update vector 1
    quiver1.set_UVC(v1_x[frame], v1_y[frame])
    
    # Update vector 2
    quiver2.set_UVC(v2_x[frame], v2_y[frame])
    
    # Update time display
    time_text.set_text(f'Time: {time[frame]:.2f} s')
    
    return quiver1, quiver2, time_text

# Create animation
# interval is in milliseconds (100 ms = 0.1s to match dt)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(v1_x), interval=100,
                              blit=True, repeat=True)

# To save the animation (optional - uncomment if needed)
# anim.save('vector_animation.gif', writer='pillow', fps=10)
# anim.save('vector_animation.mp4', writer='ffmpeg', fps=10)

plt.tight_layout()
plt.show()