# import statements
from ode_functions import *


# set parameters needed to solve ODE
t0 = 0
t1 = 50
h = 0.05
y0 = np.array([0., 2.])
gravity = 9.81
drag = 0.75
gamma = 8.0
mass = 67.0

# Butcher tableau: Classic RK4 explicit method
rk_alpha = np.array([1./6., 1./3., 1./3., 1./6.])
rk_beta = np.array([0., 1./2., 1./2., 1.])
rk_gamma = np.array([[0., 0., 0., 0.], [1./2., 0., 0., 0.], [0., 1./2., 0., 0.], [0., 0., 1., 0.]])

# Selected cord parameters
cord_length = 21  # Example SHORT cord with length 21m
cord_spring_constant = 90  # Example Spring constant for the selected cord 90


# Solve the bungee jump problem for the selected cord
t, y = explicit_solver_fixed_step(
    derivative_bungy, y0, t0, t1, h, rk_alpha, rk_beta, rk_gamma,
    gravity, cord_length, mass, drag, cord_spring_constant, gamma
)

# Calculate the maximum vertical displacement
max_vertical_displacement = np.max(y[:, 0])

# Print the results
print(f"Selected Cord: SHORT{cord_spring_constant}")
print(f"Maximum Vertical Displacement: {max_vertical_displacement} meters")



# List of cord parameters to compare
cord_parameters = [
    (16,50), 
    (16,60),
    (16,70),
    (16,80),
    (16,90),
    (16,100),
    (21,50), 
    (21,60),
    (21,70),
    (21,80),
    (21,90),
    (21,100),  
]

fully_dunked_threshold = 44.8  # Threshold value for fully dunked

max_displacements = []  # List to store maximum vertical displacements

for cord_length, cord_spring_constant in cord_parameters:
    # Run the simulation for the current cord parameters
    t, y = explicit_solver_fixed_step(
        derivative_bungy, y0, t0, t1, h, rk_alpha, rk_beta, rk_gamma,
        gravity, cord_length, mass, drag, cord_spring_constant, gamma
    )
    
    # Calculate the maximum vertical displacement
    max_vertical_displacement = np.max(y[:, 0])
    max_displacements.append(max_vertical_displacement)

# Create a plot to compare maximum vertical displacements
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cord_parameters) + 1), max_displacements, marker='o', linestyle='-')
plt.axhline(y=fully_dunked_threshold, color='r', linestyle='--', label='Fully Dunked Threshold')
plt.xlabel("Bungy Cord")
plt.ylabel("Maximum Vertical Displacement (m)")
plt.title("Comparison of Bungy Cord Performance")
plt.xticks(range(1, len(cord_parameters) + 1), [f"Cord {i}" for i in range(1, len(cord_parameters) + 1)])
plt.grid(True)
plt.legend()
plt.show()



# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, y[:, 0])
plt.xlabel("Time (s)")
plt.ylabel("Vertical Displacement (m)")
plt.title("Bungy Jump: Vertical Displacement vs. Time")

plt.subplot(2, 1, 2)
plt.plot(t, y[:, 1])
plt.xlabel("Time (s)")
plt.ylabel("Vertical Velocity (m/s)")
plt.title("Bungy Jump: Vertical Velocity vs. Time")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(y[:, 0], y[:, 1])
plt.xlabel("Vertical Displacement (m)")
plt.ylabel("Vertical Velocity (m/s)")
plt.title("Bungy Jump Phase Plot")
plt.grid(True)
plt.show()



'''
According to the graph comparing the maximum vertical displacements of various bungy cords, 
Cord 7 (with a cord length of 21 meters and a spring constant of 50) and Cord 8 
(with a cord length of 21 meters and a spring constant of 60) are the most effective that performs the 
alternatives to a great extent. It offers the most vertical displacement and breaks through the fully 
dunked threshold. Implying that it facilitates the most thrilling bungy jump possible for the jumper.'''

'''
By looking at the charts, we can get an idea of when and at which velocity the jumper hits the water. 
Approximately 2 seconds after jumping, the jumper appears to hit the water at a speed of 18 m/s.
'''

'''
The jumper's maximum vertical displacement will be reduced as a result of the additional mass. 
In order to accommodate the additional weight, the bungy cord will need to stretch even further. 
As a result, the jumper will be able to reach the water's surface faster, have a slower impact velocity, 
and take longer to do so.
'''