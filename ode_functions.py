# import statements
import numpy as np
from matplotlib import pyplot as plt


import numpy as np

def explicit_solver_fixed_step(func, y0, t0, t1, h, alpha, beta, gamma, *args):
    """
    Compute solution(s) to ODE(s) using any explicit RK method with fixed step size.

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        y0 (ndarray): initial condition(s) for dependent variable(s).
        t0 (float): start value of independent variable.
        t1 (float):	stop value of independent variable.
        h (float): fixed step size along independent variable.
        alpha (ndarray): weights in the Butcher tableau.
        beta (ndarray): nodes in the Butcher tableau.
        gamma (ndarray): RK matrix in the Butcher tableau.
        *args : optional system parameters to pass to the derivative function.

    Returns:
        t (ndarray): independent variable values at which dependent variable(s) are calculated.
        y (ndarray): dependent variable(s) solved at t values.
    """
    num_steps = int((t1 - t0) / h) + 1  # Number of steps to reach t1
    t = np.linspace(t0, t1, num_steps)  # Array of time values
    y = np.zeros((num_steps, len(y0)))  # Array to store dependent variable values

    # Initialize y0
    y[0] = y0

    # Perform time-stepping using the explicit RK method
    for i in range(1, num_steps):
        ti = t[i-1]
        yi = y[i-1]
        k_values = np.zeros((len(alpha), len(y0)))

        for j in range(len(alpha)):
            tj = ti + beta[j] * h
            sum_term = np.zeros(len(y0))

            for m in range(len(alpha)):
                sum_term += gamma[j, m] * k_values[m]

            k_values[j] = func(tj, yi + h * sum_term, *args)

        y[i] = yi + h * np.sum(alpha[:, np.newaxis] * k_values, axis=0)

    return t, y  # Return t and y values



def dp_solver_adaptive_step(func, y0, t0, t1, atol, *args):
    """
    Compute solution to ODE using the Dormand-Prince embedded RK method with an adaptive step size.

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        y0 (ndarray): initial conditions for each solution variable.
        t0 (float): start value of independent variable.
        t1 (float): stop value of independent variable.
        atol (float): error tolerance for determining adaptive step size.
        *args : optional system parameters to pass to derivative function.

    Returns:
        t (ndarray): independent variable values at which dependent variable(s) calculated.
        y (ndarray): dependent variable(s).
    """
    t_values = [t0]
    y_values = [y0]
    h = 0.001  # Initial step size
    
    while t_values[-1] < t1:
        t_current = t_values[-1]
        y_current = y_values[-1]
        h = min(h, t1 - t_current)  # Ensure we don't go past t1
        
        # Dormand-Prince coefficients
        c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1
        
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        
        b1, b2, b3, b4, b5, b6 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
        b1star, b2star, b3star, b4star, b5star, b6star = 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100
        
        # Compute k values
        k1 = h * func(t_current, y_current, *args)
        k2 = h * func(t_current + c2 * h, y_current + a21 * k1, *args)
        k3 = h * func(t_current + c3 * h, y_current + a31 * k1 + a32 * k2, *args)
        k4 = h * func(t_current + c4 * h, y_current + a41 * k1 + a42 * k2 + a43 * k3, *args)
        k5 = h * func(t_current + c5 * h, y_current + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, *args)
        k6 = h * func(t_current + c6 * h, y_current + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, *args)
        
        # Calculate y_next using the Dormand-Prince method
        y_next = y_current + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
        y_next_star = y_current + b1star * k1 + b2star * k2 + b3star * k3 + b4star * k4 + b5star * k5 + b6star * k6
        
        # Estimate the error
        error = np.linalg.norm(y_next - y_next_star)
        
        # Adjust the step size
        h = h * min(max(0.84 * (atol / error) ** 0.25, 0.1), 4.0)
        
        # Update the time and solution values
        t_values.append(t_current + h)
        y_values.append(y_next)
    
    # Convert the lists to NumPy arrays and return the results
    return np.array(t_values), np.array(y_values)





def derivative_bungy(t, y, gravity, length, mass, drag, spring, gamma):
    """
    Compute the derivatives of the bungy jumper motion.

    Args:
        t (float): independent variable, time (s).
        y (ndarray): y[0] = vertical displacement (m), y[1] = vertical velocity (m/s).
        gravity (float): gravitational acceleration (m/s^2).
        length (float):	length of the bungy cord (m).
        mass (float): the bungy jumper's mass (m).
        drag (float): coefficient of drag (kg/m).
        spring (float): spring constant of the bungy cord (N/m).
        gamma (float): coefficient of damping (Ns/m).

    Returns:
        f (ndarray): derivatives of vertical position and vertical velocity.
    """
def derivative_bungy(t, y, gravity, length, mass, drag, spring, gamma):
    y_pos, y_vel = y  # Vertical displacement and velocity

    # Calculate derivatives
    g = gravity
    L = length
    m = mass
    cd = drag
    k = spring
    damping = gamma

    # When the cord is slack
    if y_pos <= L:
        drag_force = -np.sign(y_vel) * (cd * y_vel**2) / m
        y_acceleration = g + drag_force
    # When the cord is stretched
    else:
        drag_force = -np.sign(y_vel) * (cd * y_vel**2) / m
        spring_force = (k / m) * (y_pos - L)
        damping_force = (damping / m) * y_vel
        y_acceleration = g + drag_force - spring_force - damping_force

    return np.array([y_vel, y_acceleration])



def derivative_lorenz(t, y, sigma, rho, beta):
    """
    Compute the derivatives for the Lorenz system.

    Args:
        t (float): independent variable, time.
        y (ndarray): dependent variables x, y and z in Lorenz system.
        sigma (float): system parameter of Lorenz system.
        rho (float): system parameter of Lorenz system.
        beta (float): system parameter of Lorenz system.

    Returns:
        f (ndarray): derivatives of x, y and z in Lorenz system.
    """
    x, y, z = y  # Unpack the dependent variables

    # Compute the derivatives of x, y, and z based on the Lorenz system equations
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    # Return the derivatives as a NumPy array
    return np.array([dxdt, dydt, dzdt])

