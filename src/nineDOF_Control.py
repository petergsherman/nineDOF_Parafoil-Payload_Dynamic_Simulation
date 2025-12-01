#nineDOF_Control.py
import numpy as np

class baseController:
    """Simple altitude-based controller"""
    
    def __init__(self) -> None:
        self.deltaL = 0.0
        self.deltaR = 0.0
        self.incidence = 0.0

    def computeControl(self, state):
        """
        state[2] is altitude (negative = above ground)
        t is the current time
        Returns: (deltaL, deltaR, incidence)
        """
        altitude = -state[2]  # Convert to positive altitude
        
        if altitude > 50:
            # High altitude: apply right brake to start turning
            return (0.0, 0.9, 0.0)
        elif altitude > 25:
            # Medium altitude: reduce braking
            return (0.0, 0.5, 0.0)
        else:
            # Low altitude: no control
            return (0.0, 0.0, 0.0)


# Example usage wrapper for plant.run_simulation()
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