#nineDOF_Plant.py
import numpy as np
from typing import Tuple

from nineDOF_Parameters import systemParameters, atmosphereParameters
from nineDOF_Transform import makeJ, makeTransform_IntertialToBody

class plant:

    def __init__(self, parameters: systemParameters, atmosphere: atmosphereParameters) -> None:
        self.parameters = parameters
        self.atmosphere = atmosphere

        #Populate Mass and Inertia Tensors
        self.populateInertias()

    def populateInertias(self):
        p = self.parameters

        #Parafoil Inertia Matrix
        self.I_P = np.array([
            [p.PIXX, p.PIXY, p.PIXZ],
            [p.PIXY, p.PIYY, p.PIYZ],
            [p.PIXZ, p.PIYZ, p.PIZZ]
        ])
        
        #Cradle inertia matrix
        self.I_C = np.array([
            [p.CIXX, p.CIXY, p.CIXZ],
            [p.CIXY, p.CIYY, p.CIYZ],
            [p.CIXZ, p.CIYZ, p.CIZZ]
        ])
        
        #Apparent mass matrix (diagonal)
        self.I_AM = np.diag([p.PMASSA, p.PMASSB, p.PMASSÇ])
        
        #Apparent inertia matrix (diagonal)
        self.I_AI = np.diag([p.PMASSP, p.PMASSQ, p.PMASSR])
        
        #Spanwise camber matrix (off-diagonal coupling)
        self.I_H = np.array([
            [0.0, p.PMASSH, 0.0],
            [p.PMASSH, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
    
    def makeT_PPI(self, incidence, nominalIncidence):
        
        total_incidence = incidence + self.parameters.NOM_INCIDENCE
        ci, si = np.cos(total_incidence), np.sin(total_incidence)
        T_PPI = np.array([
            [ci, 0, si],
            [0,  1,  0],
            [-si,0, ci]
        ])
        return T_PPI

    def skew(self, v: np.ndarray) -> np.ndarray:
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def interpolateAeroTables(self, sig1: float) -> Tuple[float, float, float, float]:
        """Interpolate CD0, CDA2, CL0, CLA from sig1 table"""
        p = self.parameters
        
        # Find bracket
        if sig1 <= p.SIGTAB[0]:
            return p.CD0TAB[0], p.CDA2TAB[0], p.CL0TAB[0], p.CLATAB[0]
        elif sig1 >= p.SIGTAB[-1]:
            return p.CD0TAB[-1], p.CDA2TAB[-1], p.CL0TAB[-1], p.CLATAB[-1]
        
        # Linear interpolation
        idx = np.searchsorted(p.SIGTAB, sig1) - 1
        idx = np.clip(idx, 0, len(p.SIGTAB)-2)
        
        m = (sig1 - p.SIGTAB[idx]) / (p.SIGTAB[idx+1] - p.SIGTAB[idx])
        
        cd0 = p.CD0TAB[idx] + m*(p.CD0TAB[idx+1] - p.CD0TAB[idx])
        cda2 = p.CDA2TAB[idx] + m*(p.CDA2TAB[idx+1] - p.CDA2TAB[idx])
        cl0 = p.CL0TAB[idx] + m*(p.CL0TAB[idx+1] - p.CL0TAB[idx])
        cla = p.CLATAB[idx] + m*(p.CLATAB[idx+1] - p.CLATAB[idx])
        
        return cd0, cda2, cl0, cla
    
    def interpolateCND2(self, alpha: float) -> float:
        """Interpolate CND2 from AOA table"""
        p = self.parameters
        
        if alpha <= p.AOATAB[0]:
            return p.CND2TAB[0]
        elif alpha >= p.AOATAB[-1]:
            return p.CND2TAB[-1]
        
        idx = np.searchsorted(p.AOATAB, alpha) - 1
        idx = np.clip(idx, 0, len(p.AOATAB)-2)
        
        m = (alpha - p.AOATAB[idx]) / (p.AOATAB[idx+1] - p.AOATAB[idx])
        
        return p.CND2TAB[idx] + m*(p.CND2TAB[idx+1] - p.CND2TAB[idx])

    def computeAerodynamics(self, state: np.ndarray, deltaL: float, deltaR: float, incidence: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.parameters

        #Extract State Components
        phiP, thetaP, psiP = state[3:6]
        phiC, thetaC, psiC = state[6:9]
        uG, vG, wG = state[9:12]
        pP, qP, rP = state[12:15]
        pC, qC, rC = state[15:18]

        #Transformation Matrices
        T_IP = makeTransform_IntertialToBody(phiP, thetaP, psiP)
        T_IC = makeTransform_IntertialToBody(phiC, thetaC, psiC)
        T_PC = T_IP.T @ T_IC

        #Velocities of Gimbal in parafoil frame
        vG_P = np.array([uG, vG, wG])
        wP_P = np.array([pP, qP, rP])
        wC_C = np.array([pC, qC, rC])
        S_wP_P = self.skew(wP_P)
        S_wC_C = self.skew(wC_C)

        #Geometry Vectors
        rPG_P = np.array([-p.SLPARAFOIL, -p.BLPARAFOIL, -p.WLPARAFOIL])
        rPR_P = np.array([p.SLPR, p.BLPR, p.WLPR])
        rRAp_PI = np.array([p.SLRAP, p.BLRAP, p.WLRAP])

        rGC_C = np.array([p.SLCRADLE, p.BLCRADLE, p.WLCRADLE])

        #Incidence Rotation Matrix
        T_PPI = self.makeT_PPI(incidence, p.NOM_INCIDENCE)
            
        #Rotation Point to Aero Center in Parafoil Frame
        rRAp_P = T_PPI @ rRAp_PI
        rPAp_P = rPR_P + rRAp_P
        rGAp_P = -rPG_P + rPAp_P

        #Velocity of the Aerodynamic Center
        vAp_P = vG_P + S_wP_P @ rGAp_P
        vAp_P_wind = vAp_P - np.array([self.atmosphere.VXWIND,self.atmosphere.VYWIND, self.atmosphere.VZWIND])
        vAp_PI = T_PPI.T @ vAp_P_wind #Rotation by incidence frame
        vAp = np.linalg.norm(vAp_PI)

        #Computing Cradle Velocity
        vC_C = T_PC @ vG_P + S_wC_C @ rGC_C
        vC = np.linalg.norm(vC_C)

        #Avoiding Singularity Condition
        if vAp < 0.01: 
            return np.zeros(3), np.zeros(3), np.zeros(3)
        
        #Calculating Angle of Attack and Sideslip
        alpha = np.arctan2(vAp_PI[2], vAp_PI[0])
        beta = np.arcsin(np.clip(vAp_PI[1] / vAp, -1, 1))

        #Computing Controls
        dA = deltaR - deltaL
        dS = 0.5 * (deltaR + deltaL) - p.deadband + p.NOM_BRAKE
        sig1 = dS / p.dbar

        #Interpolate Aerodynamic Coefficients
        CD0, CDA2, CL0, CLA = self.interpolateAeroTables(sig1)
        CND2 = self.interpolateCND2(alpha)

        #Normalized Independent Controls
        sigmaLeft = deltaL / p.dbar
        sigmaRight = deltaR / p.dbar

        #Aerodynamic Force and Moment Coefficients
        Cd = CD0 + CDA2 * (alpha ** 2)
        Cy = p.CYBETA * beta
        Cl = CL0 + (CLA * alpha)

        C_l = (p.CLBETA*beta + (p.bbar_parafoil/(2*vAp))*p.CLP*pP + 
               (p.bbar_parafoil/(2*vAp))*p.CLR*rP + p.CLDA*dA/p.dbar)
        C_m = p.CM0 + (p.cbar_parafoil/(2*vAp))*p.CMQ*qP + p.CMDS*dS/p.dbar
        C_n = (p.CNBETA*beta + (p.bbar_parafoil/(2*vAp))*p.CNP*pP + 
               (p.bbar_parafoil/(2*vAp))*p.CNR*rP - p.CND1*sigmaLeft - 
               CND2*sigmaLeft*sig1 + p.CND1*sigmaRight + CND2*sigmaRight*sig1)

        #Stall Model
        Cl = np.clip(Cl, 0.5, 2.0)
        Cd = np.clip(Cd, 0,   1.5)

        #Calculating Forces and Moments
        C_FAp_PI = 0.5 * self.atmosphere.DEN * p.A_parafoil * (vAp ** 2) * np.array([-(np.cos(alpha) * Cd) + (np.sin(alpha) * Cl), #Aerodynamic Force on Parafoil in the Incidence Frame
                                                                                       Cy,
                                                                                     (-np.sin(alpha) * Cd) - (np.cos(alpha) * Cl)]) 

        C_FAp_P = T_PPI @ C_FAp_PI #Rotating into Parafoil Frame

        C_MAp_P = 0.5 * self.atmosphere.DEN * p.A_parafoil * (vAp ** 2) * np.array([p.bbar_parafoil * C_l,
                                                                                    p.cbar_parafoil * C_m,
                                                                                    p.bbar_parafoil * C_n])

        C_FAc_C = -0.5 * self.atmosphere.DEN * vC * p.A_cradle * p.CD_cradle * vC_C

        return C_FAp_P, C_MAp_P, C_FAc_C

    def computeDerivatives(self, state: np.ndarray, deltaL: float, deltaR: float, incidence: float) -> np.ndarray:
        p = self.parameters
        statedot = np.zeros(18)

        #Extract State Vector
        xG, yG, zG = state[0:3]
        phiP, thetaP, psiP = state[3:6]
        phiC, thetaC, psiC = state[6:9]
        uG, vG, wG = state[9:12]
        pP, qP, rP = state[12:15]
        pC, qC, rC = state[15:18]

        #Vectors for Readability
        vG_P = np.array([uG, vG, wG])
        wP_P = np.array([pP, qP, rP])
        wC_C = np.array([pC, qC, rC])

        #Skew of Angular Velocities
        S_wP_P = self.skew(wP_P)
        S_wC_C = self.skew(wC_C)

        #Creating Transformation Matrices
        T_IP = makeTransform_IntertialToBody(phiP, thetaP, psiP)
        T_IC = makeTransform_IntertialToBody(phiC, thetaC, psiC)
        T_CP = T_IC.T @ T_IP
        T_PPI = self.makeT_PPI(incidence, p.NOM_INCIDENCE)
        T_parafoilJ = makeJ(phiP, thetaP, psiP)
        T_bodyJ = makeJ(phiC, thetaC, psiC)

        #Updating Non-coupled Linear Statedot Terms
        statedot[0:3] = T_IP @ vG_P
        statedot[3:6] = T_parafoilJ @ wP_P
        statedot[6:9] = T_bodyJ @ wC_C

        #Setup geometry vectors
        rPG_P = np.array([-p.SLPARAFOIL, -p.BLPARAFOIL, -p.WLPARAFOIL])
        rGC_C = np.array([p.SLCRADLE, p.BLCRADLE, p.WLCRADLE])
        rPMp_P = np.array([p.SLPMP, p.BLPMP, p.WLPMP])
        rGMp_P = -rPG_P + rPMp_P

        rPR_P = np.array([p.SLPR, p.BLPR, p.WLPR])
        rRAp_PI = np.array([p.SLRAP, p.BLRAP, p.WLRAP])
        rRAp_P = T_PPI @ rRAp_PI
        rPAp_P = rPR_P + rRAp_P

        #Skew of Position Vectors
        S_rCG_C = self.skew(-rGC_C)
        S_rPG_P = self.skew(rPG_P)
        S_rPMp_P = self.skew(rPMp_P)
        S_rGMp_P = self.skew(rGMp_P)
        S_rPAp_P = self.skew(rPAp_P)

        #Velocity of Apparent Mass Center
        vMp_P = vG_P + S_wP_P @ rGMp_P
        S_vMp_P = self.skew(vMp_P)

        #Getting Aerodynamic Forces
        C_FAp_P, C_MAp_P, C_FAc_C = self.computeAerodynamics(state, deltaL, deltaR, incidence)

        #Computing Gravitational Forces
        C_FWp_P = p.m_parafoil * p.GRAVITY * np.array([-np.sin(thetaP),
                                                        np.sin(phiP) * np.cos(thetaP),
                                                        np.cos(phiP) * np.cos(thetaP)])

        C_FWc_C = p.m_cradle * p.GRAVITY * np.array([-np.sin(thetaC),
                                                      np.sin(phiC) * np.cos(thetaC),
                                                      np.cos(phiC) * np.cos(thetaC)])

        #Computing Gimbal Stiffness Moment
        C_MG_P = np.array([0, 
                           0, 
                           p.KGIMBAL * (psiP - psiC) + p.CGIMBAL * (statedot[5] - statedot[8])])

        #Computing A matrix for Ax=b
        A11 = p.m_cradle * S_rCG_C
        A12 = np.zeros((3, 3))
        A13 = p.m_cradle * T_IC.T @ T_IP
        A14 = -T_IC.T

        A21 = np.zeros((3, 3))
        A22 = -self.I_AM @ S_rGMp_P + self.I_H + p.m_parafoil * S_rPG_P
        A23 = p.m_parafoil * np.eye(3) + self.I_AM
        A24 = T_IP.T

        A31 = self.I_C
        A32 = np.zeros((3, 3))
        A33 = np.zeros((3, 3))
        A34 = -S_rCG_C @ T_IC.T

        A41 = np.zeros((3, 3))
        A42 = self.I_P + S_rPMp_P @ (self.I_H - self.I_AM @ S_rGMp_P) - self.I_H @ S_rGMp_P + self.I_AI
        A43 = self.I_H + S_rPMp_P @ self.I_AM
        A44 = S_rPG_P @ T_IP.T

        A = np.block([[A11, A12, A13, A14],
                      [A21, A22, A23, A24],
                      [A31, A32, A33, A34],
                      [A41, A42, A43, A44]])

        #Making b Vector for Ax = b
        b1 = C_FAc_C + C_FWc_C - p.m_cradle * T_CP @ S_wP_P @ vG_P - p.m_cradle * S_wC_C @ S_wC_C @ rGC_C
        b2 = C_FAp_P + C_FWp_P - p.m_parafoil * S_wP_P @ vG_P - S_wP_P @ (self.I_AM @ vMp_P + self.I_H @ wP_P) + p.m_parafoil * S_wP_P @ S_wP_P @ rPG_P
        b3 = T_CP @ C_MG_P - S_wC_C @ self.I_C @ wC_C
        b4 = C_MAp_P + S_rPAp_P @ C_FAp_P - C_MG_P - (S_rPMp_P @ S_wP_P + S_vMp_P @ self.I_H + S_wP_P @ (self.I_AI + self.I_P)) @ wP_P - (S_rPMp_P @ S_wP_P @ self.I_AM + S_wP_P @ self.I_H) @ vMp_P
        b = np.concatenate([b1, b2, b3, b4])

        #Solving Linear Algebra Ax = b
        try: 
            X = np.linalg.solve(A,b)
        except np.linalg.LinAlgError:
            print("Singular Matrix in Dynamics Computation")
            X = np.zeros(12)
        
        #Populating Statedot with Nonlinear Coupled Terms
        statedot[9:12]  = X[6:9]
        statedot[12:15] = X[3:6]
        statedot[15:18] = X[0:3]

        return statedot

    def rk4_step(self, state: np.ndarray, dt: float, deltaL: float, deltaR: float, incidence: float) -> np.ndarray:
        k1 = dt * self.computeDerivatives(state, deltaL, deltaR, incidence)
        k2 = dt * self.computeDerivatives(state + k1/2, deltaL, deltaR, incidence)
        k3 = dt * self.computeDerivatives(state + k2/2, deltaL, deltaR, incidence)
        k4 = dt * self.computeDerivatives(state + k3, deltaL, deltaR, incidence)
        
        return state + (k1 + 2*k2 + 2*k3 + k4) / 6

    def run_simulation(self, state0: np.ndarray, t_final: float, dt: float,
                      control_func=None) -> Tuple[np.ndarray, np.ndarray]:

        n_steps = int(t_final / dt) + 1
        times = []
        states = []
        
        state = state0.copy()
        times.append(0.0)
        states.append(state)
        
        for i in range(1, n_steps):
            t = times[i-1]
            
            # Update atmosphere (assuming z is state[2], positive down)
            if hasattr(self.atmosphere, "update"):
                altitude = -states[i - 1][2]
                print(altitude)
                airspeed = np.linalg.norm([states[i-1][9:12]])
                self.atmosphere.update(t, altitude, airspeed)

            #Get control inputs
            if control_func is not None:
                deltaL, deltaR, incidence = control_func(t, states[i-1])
            else:
                # Default: nominal controls
                deltaL = 0.0
                deltaR = 0.0
                incidence = 0.0
            
            #Integrate
            state = self.rk4_step(states[i-1], dt, deltaL, deltaR, incidence)
            
            #Angle wrapping
            for angle_idx in [4, 5, 7, 8]:  #theta and psi for both bodies
                while state[angle_idx] > np.pi:
                    state[angle_idx] -= 2*np.pi
                while state[angle_idx] < -np.pi:
                    state[angle_idx] += 2*np.pi
            
            times.append(t + dt)
            states.append(state.copy())
            
            if state[2] >= 0.0:
                break
        
        return np.array(times), np.array(states)
