#nineDOF_Control.py
import numpy as np

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