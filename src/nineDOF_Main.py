#nineDOF_Main.py
import numpy as np
from nineDOF_Plant import plant
from nineDOF_Control import simpleHeadingController, testController, make_control_function
from nineDOF_Parameters import systemParameters, atmosphereParameters
from nineDOF_Visualization import visualizeData


#Create parameters from input file
params = systemParameters()
atm = atmosphereParameters()
    
#Create simulator
sim = plant(params, atm)
    
#Initial state: hovering at origin with small perturbation
state0 = np.zeros(18)
state0[0:3] = [0.0, 0.0, -100.0]  # Position: 100m altitude
state0[3:6] = [0.0, 0.1, 0.0]     # Parafoil angles: small pitch
state0[6:9] = [0.0, 0.1, 0.0]     # Cradle angles: small pitch
state0[9:12] = [10.0, 0.0, -0.5]  # Forward velocity
    
#Run simulation
print("Running parafoil-payload simulation...")
t_final = 50.0  # seconds
dt = 0.1        # seconds
targetLandingPoint = (1000.0, 1000.0) #X and Y of the targeted Landing Point
    
#Creating Control Function
control = make_control_function(simpleHeadingController(targetLandingPoint))

times, states = sim.run_simulation(state0, t_final, dt, control) #0.94 is maximum control defelction 
    
print(f"Simulation complete: {len(times)} time steps")
print(f"Final position: x={states[-1,0]:.2f}, y={states[-1,1]:.2f}, z={states[-1,2]:.2f}")
print(f"Final velocity: u={states[-1,9]:.2f}, v={states[-1,10]:.2f}, w={states[-1,11]:.2f}")
    
#Simple visualization
visualizeData.plot_trajectory(states)