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
        
        if altitude > 60:
            return (0.0, 0.0, 0.0)
        elif altitude < 60:
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
        T_IP = makeTransform_IntertialToBody(state[3], state[4], state[5]) #Transformation from Inertial to Parafoil Frame

        #Calculating Parafoil Heading Unit Vector 
        heading_P = np.array([state[9], state[10], 0]) / np.linalg.norm(np.array([state[9], state[10], 0])) 
        heading_I = T_IP.T @ heading_P #Parafoil to Inertial

        #Calculating Inertial Heading Unit Vector
        headingTarget_I = np.array([self.target[0] - state[0], self.target[1] - state[1], 0]) / np.linalg.norm(np.array([self.target[0] - state[0], self.target[1] - state[1], 0]))
        
        #Computing Cross Product of Heading and Target Heading
        cross = np.cross(heading_I[0:2], headingTarget_I[0:2])
        
        return self.asymmetric_to_brakes(cross, 0.5)
        
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