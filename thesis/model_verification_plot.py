
import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV files (no headers, 3 columns: x, y, z)
left_turn = pd.read_csv('left_turn.csv', header=None, names=['x', 'y', 'z'])
right_turn = pd.read_csv('right_turn.csv', header=None, names=['x', 'y', 'z'])
no_deflection = pd.read_csv('no_deflection.csv', header=None, names=['x', 'y', 'z'])
symmetric_braking = pd.read_csv('symmetric_braking.csv', header=None, names=['x', 'y', 'z'])



# Plot 1: Top-down view (Y-X) of left and right turns
plt.figure(figsize=(10, 8))
plt.plot(left_turn['y'], left_turn['x'], 'b-', linewidth=2, label='Left Turn')
plt.plot(right_turn['y'], right_turn['x'], 'r-', linewidth=2, label='Right Turn')
plt.xlabel('Y Position (m)', fontsize=12)
plt.ylabel('X Position (m)', fontsize=12)
plt.title('Top-Down View: Turning Maneuvers', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('turning_maneuvers.png', dpi=300, bbox_inches='tight')
print("Turning maneuvers plot saved as 'turning_maneuvers.png'")
plt.show()

# Plot 2: Side view (X-Z) of no deflection and symmetric braking
# Note: z is negative altitude, so we'll plot -z to show altitude
plt.figure(figsize=(10, 8))
plt.plot(no_deflection['x'], -no_deflection['z'], 'g-', linewidth=2, label='No Deflection')
plt.plot(symmetric_braking['x'], -symmetric_braking['z'], 'm-', linewidth=2, label='Symmetric Braking')
plt.xlabel('X Position (m)', fontsize=12)
plt.ylabel('Altitude (m)', fontsize=12)
plt.title('Side View: Forward Flight Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forward_flight.png', dpi=300, bbox_inches='tight')
print("Forward flight plot saved as 'forward_flight.png'")
plt.show()