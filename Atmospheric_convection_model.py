# import statements
from ode_functions import *

# Lorenz system parameters
sigma = 10.
rho = 28.
beta = 8. / 3.

# Initial conditions 
y0 = np.array([1.0, 1.0, 1.0]) # Initial conditions: x(0) = 1.0, y(0) = 1.0, z(0) = 1.0

# Time parameters
t0 = 0.0
t1 = 40.0 

# Error tolerance 
error_tol = 1e-3  # Increased error tolerance for faster execution rather than 1e-5

# Initial time step
h = 1e-3

# Solve the Lorenz system using the Dormand-Prince method
t, y = dp_solver_adaptive_step(derivative_lorenz, y0, t0, t1, error_tol, sigma, rho, beta)


# Plot the results - Phase Plot (xz-plane)
plt.figure(figsize=(10, 6))
plt.plot(y[:, 0], y[:, 2])
plt.xlabel('x')
plt.ylabel('z')
plt.title('Lorenz Attractor (Phase Plot)')
plt.grid(True)
plt.show()


# Plot the results - Timeseries Plot
plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label='x')
plt.plot(t, y[:, 1], label='y')
plt.plot(t, y[:, 2], label='z')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Lorenz Attractor (Timeseries)')
plt.legend()
plt.grid(True)
plt.show()


# Initial conditions - First set
y0_first = np.array([1.0, 1.0, 1.0])  # Initial conditions: x(0) = 1.0, y(0) = 1.0, z(0) = 1.0

# Initial conditions - Second set
y0_second = np.array([1.001, 1.001, 1.001])  # New initial conditions: x(0) = 1.001, y(0) = 1.001, z(0) = 1.001

# Solve the Lorenz system using the Dormand-Prince method - First set of initial conditions
t_first, y_first = dp_solver_adaptive_step(derivative_lorenz, y0_first, t0, t1, error_tol, sigma, rho, beta)

# Solve the Lorenz system using the Dormand-Prince method - Second set of initial conditions
t_second, y_second = dp_solver_adaptive_step(derivative_lorenz, y0_second, t0, t1, error_tol, sigma, rho, beta)

# Plot the results - Phase Plot (xz-plane) - Overlay both sets of results
plt.figure(figsize=(10, 6))
plt.plot(y_first[:, 0], y_first[:, 2], label='Initial conditions 1')
plt.plot(y_second[:, 0], y_second[:, 2], label='Initial conditions 2')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Lorenz Attractor (Phase Plot)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the results - Timeseries Plot - Overlay both sets of results
plt.figure(figsize=(10, 6))
plt.plot(t_first, y_first[:, 0], label='x (Initial conditions 1)')
plt.plot(t_first, y_first[:, 1], label='y (Initial conditions 1)')
plt.plot(t_first, y_first[:, 2], label='z (Initial conditions 1)')
plt.plot(t_second, y_second[:, 0], linestyle='--', label='x (Initial conditions 2)')
plt.plot(t_second, y_second[:, 1], linestyle='--', label='y (Initial conditions 2)')
plt.plot(t_second, y_second[:, 2], linestyle='--', label='z (Initial conditions 2)')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Lorenz Attractor (Timeseries)')
plt.legend()
plt.grid(True)
plt.show()



'''
The Lorenz system never reaches a stable state and is chaotic by definition.
Instead, it demonstrates a unusual attractor—a complex, non-repeating behavior.
The system's chaotic and irregular oscillations, in which it continuously explores 
new states and does not converge to a single point, are depicted by the phase plot 
(Lorenz butterfly) and the timeseries plot.

In conclusion, the Lorenz system is a typical illustration of a chaotic system that exhibits sensitivity 
to initial conditions and lacks a steady-state or long-term equilibrium.
'''

'''
When you use a second set of initial conditions (x(0) = y(0) = z(0) = 1.001) in Step 6 to solve the 
Lorenz system, you will notice that the solution differs from the original. The first set of initial 
conditions (x(0) = y(0) = z(0) = 1.0) led to the solution over time. Chaotic systems' extreme sensitivity 
is demonstrated by this behavior. To beginning conditions, like the Lorenz framework.

It is essential to keep in mind, in terms of numerical accuracy, that chaotic systems are fundamentally 
unpredictable in the long run due to their sensitivity to initial conditions. Long-term outcomes could 
be vastly different from one another even with relatively minor variations in starting conditions. In a 
chaotic system, predicting the precise values of x, y, and z at any given time is extremely challenging 
due to this sensitivity.

numerical errors in chaotic systems can aggregate and intensify over the long term, making long-term 
gauges really sketchy. Therefore, it is frequently impossible to accurately predict the precise values 
of the variables in chaotic systems. Instead, the qualitative behavior of attractors like the Lorenz 
attractor and statistical characteristics of chaotic systems are frequently the focus of research.      
'''