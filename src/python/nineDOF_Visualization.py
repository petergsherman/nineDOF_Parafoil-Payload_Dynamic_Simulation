import numpy as np
import matplotlib.pyplot as plt

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

        # Mark Start (Green Dot)
        ax.scatter(east[0], north[0], altitude[0], c='green', s=50, marker='.', label='Start')
        
        # Find ground impact point (where altitude <= 0)
        ground_impact_idx = np.where(altitude <= 0)[0]
        if len(ground_impact_idx) > 0:
            # Use first point where altitude <= 0
            impact_idx = ground_impact_idx[0]
        else:
            # If no point crosses zero, find the point closest to zero
            impact_idx = np.argmin(np.abs(altitude))
        
        # Mark Ground Impact (Red X)
        ax.scatter(east[impact_idx], north[impact_idx], altitude[impact_idx], c='red', s=100, marker='x', label='Ground Impact')

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
    
    def plot_atmosphere(atm):
        """
        Two-subplot atmosphere visualization:

        Top:
            Wind speed vs altitude (layered mean wind profile)

        Bottom:
            Gust direction (deg, 0 to 360) vs time on left axis,
            Gust speed (m/s) vs time on right axis.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

        # ===============================
        # TOP: Wind speed vs altitude
        # ===============================
        if hasattr(atm, "get_wind_profile"):
            altitudes, speeds, _dirs_deg = atm.get_wind_profile()
        else:
            # Fallback if method not present
            altitudes = np.array(atm.layer_altitudes, dtype=float)
            speeds = np.array(atm.layer_speeds, dtype=float)

        ax1.plot(speeds, altitudes)
        ax1.set_xlabel("Wind Speed [m/s]")
        ax1.set_ylabel("Altitude [m]")
        ax1.set_title("Atmospheric Wind Layering")
        ax1.grid(True)

        # ===============================
        # BOTTOM: Gust direction & speed vs time
        # ===============================
        if not (hasattr(atm, "hist_t") and hasattr(atm, "hist_gust_dir_deg") and hasattr(atm, "hist_gust_speed")):
            ax2.text(0.5, 0.5,
                    "No gust history found.\nAdd logging in dynamicAtmosphere.update().",
                    ha="center", va="center", transform=ax2.transAxes)
            ax2.set_axis_off()
            plt.tight_layout()
            plt.show()
            return

        t = np.array(atm.hist_t, dtype=float)
        gust_dir_raw = np.array(atm.hist_gust_dir_deg, dtype=float)
        gust_speed = np.array(atm.hist_gust_speed, dtype=float)

        # Convert angles from [-180, 180] to [0, 360] if needed
        # Then unwrap to prevent discontinuities
        gust_dir = gust_dir_raw.copy()
        
        # First convert any negative angles to 0-360 range
        gust_dir = np.where(gust_dir < 0, gust_dir + 360, gust_dir)
        
        # Unwrap the angles to prevent jumps (e.g., 359° -> 1° becomes 359° -> 361°)
        gust_dir = np.degrees(np.unwrap(np.radians(gust_dir)))
        
        # Optionally, normalize back to 0-360 range for cleaner display
        # Comment out the next line if you want to see the unwrapped continuous values
        gust_dir = gust_dir % 360

        ax2_dir = ax2
        ax2_spd = ax2.twinx()

        # Plot gust direction with color to emphasize bidirectionality
        ax2_dir.plot(t, gust_dir, color='tab:blue', linewidth=1.5, label='Gust Direction')
        
        # Plot gust speed
        ax2_spd.plot(t, gust_speed, color='tab:orange', linewidth=1.5, label='Gust Speed')

        ax2_dir.set_xlabel("Time [s]")
        ax2_dir.set_ylabel("Gust Direction [deg]", color='tab:blue')
        ax2_dir.tick_params(axis='y', labelcolor='tab:blue')
        
        # Set limits to 0-360 range
        ax2_dir.set_ylim([0, 360])
        ax2_dir.set_yticks(np.arange(0, 361, 45))

        ax2_spd.set_ylabel("Gust Speed [m/s]", color='tab:orange')
        ax2_spd.tick_params(axis='y', labelcolor='tab:orange')
        
        # Ensure gust speed starts at 0
        current_ylim = ax2_spd.get_ylim()
        ax2_spd.set_ylim([0, max(current_ylim[1], gust_speed.max() * 1.1)])

        ax2_dir.set_title("Gust Direction & Speed vs Time")
        ax2_dir.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()