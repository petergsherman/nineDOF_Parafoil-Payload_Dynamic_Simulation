#nineDOF_Transform.py
import numpy as np

def makeTransform_IntertialToBody(phi: float, theta: float, psi: float) -> np.ndarray:
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        
        T = np.array([
            [ctheta*cpsi, sphi*stheta*cpsi - cphi*spsi, cphi*stheta*cpsi + sphi*spsi],
            [ctheta*spsi, sphi*stheta*spsi + cphi*cpsi, cphi*stheta*spsi - sphi*cpsi],
            [-stheta, sphi*ctheta, cphi*ctheta]
        ])

        return T

def makeJ(phi, theta, psi): #Euler-rate Jacobian Tensor
        c_phi = np.cos(phi)
        c_theta = np.cos(theta)
        s_phi = np.sin(phi)
        t_theta = np.tan(theta)

        return np.array([[1,  s_phi * t_theta,     c_phi * t_theta],
                         [0,  c_phi,               -s_phi],
                         [0,  s_phi / c_theta,     c_phi / c_theta]])
                    
