#nineDOF_Visualization.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

class visualizeData:
    def plot_trajectory(history, title="Parafoil Trajectory"):
        # Extract positions
        # NED Coordinates: x=North, y=East, z=Down
        north = history[:, 0]
        east  = history[:, 1]
        down  = history[:, 2]
        
        # Convert 'Down' to 'Altitude' for visualization (Alt = -Down)
        altitude = -down

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the line
        ax.plot(east, north, altitude, label='Trajectory', linewidth=2)

        # Mark Start (Green Circle) and End (Red X)
        ax.scatter(east[0], north[0], altitude[0], c='green', s=50, marker='o', label='Start')
        ax.scatter(east[-1], north[-1], altitude[-1], c='red', s=100, marker='x', label='Impact/End')

        # Labels
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(title)
        
        # Force equal aspect ratio implies simpler interpretation of distance
        max_range = np.array([east.max()-east.min(), north.max()-north.min(), altitude.max()-altitude.min()]).max() / 2.0
        mid_x = (east.max()+east.min()) * 0.5
        mid_y = (north.max()+north.min()) * 0.5
        mid_z = (altitude.max()+altitude.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        plt.grid(True)
        plt.show()
    
    def plot_Atmosphere(atm):
        altitudes, speeds, directions = atm.get_wind_profile()
        plt.plot(speeds, altitudes)
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Altitude (m)')
        plt.show()