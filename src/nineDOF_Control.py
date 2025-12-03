#nineDOF_Control.py
import numpy as np
from nineDOF_Transform import makeTransform_IntertialToBody

class testController:
    def __init__(self) -> None:
        self.deltaL = 0.0
        self.deltaR = 0.0
        self.incidence = 0.0

    def computeControl(self, state):
        altitude = -state[2]  # Convert to positive altitude
        
        if altitude > 250:
            return (0.0, 0.0, 0.0)
        elif altitude < 250:
            return (0.0, 0.9, 0.0)
        else:
            return (0.0, 0.0, 0.0)

class simpleHeadingController:
    def __init__(self, targetLandingLocation) -> None:
        self.deltaL = 0.0
        self.deltaR = 0.0
        self.incidence = 0.0
        self.target = targetLandingLocation
    
    def asymmetric_to_brakes(self, ctrl, nominal=0.5):
            """Convert ctrl in [-1,1] to (trbar, tlbar) in [0,1]. Negative=left, Positive=right"""
            ctrl = np.clip(ctrl, -0.9, 0.9)
            deltaL = nominal - nominal * ctrl  # Left: increases when ctrl < 0
            deltaR = nominal + nominal * ctrl  # Right: increases when ctrl > 0
            return (np.clip(deltaL, 0, 0.9), np.clip(deltaR, 0, 0.9), 0)

    def computeControl(self, state):
        #Calculating Parafoil Heading Unit Vector 
        heading_I = np.array([np.cos(state[5]), np.sin(state[5])])

        #Calculating Inertial Heading Unit Vector
        headingTarget_I = np.array([self.target[0] - state[0], self.target[1] - state[1]]) / np.linalg.norm(np.array([self.target[0] - state[0], self.target[1] - state[1]]))
        
        #Computing Cross Product of Heading and Target Heading
        cross = np.cross(heading_I[0:2], headingTarget_I[0:2])
        print(heading_I[0], heading_I[1], headingTarget_I[0], headingTarget_I[1], sep=', ')

        if cross > 0:
            return (0, np.abs(cross), 0) #Right Break Control
        elif cross < 0:
            return (np.abs(cross), 0, 0) #Left Break Control
        
        
def make_control_function(controller):
    """
    Wraps a controller object to work with plant.run_simulation()
    
    Usage:
        controller = PDController(target_altitude=-50.0)
        control_func = make_control_function(controller)
        times, states = sim.run_simulation(state0, t_final, dt, control_func)
    """
    def control_wrapper(t, state):
        return controller.computeControl(state)
    return control_wrapper