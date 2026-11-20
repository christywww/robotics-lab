


Control.py

The trajectory is a chain of Bézier curves joined so that position, velocity and acceleration are continuous at the joins. The Bézier curves are computed by solving a quadratic program for each segment using quadprog. The cost is a least squares fit to the path plus an acceleration penalty. The curves are constrained to lie within joint limits and the trajectory starts and ends at rest.

A torque PID controller based on joint error plus feedforward acceleration generates a desired acceleration, which is converted to joint torques via RNEA (Pinocchio). A squeeze torque ensures end effectors hold the cube and force-feedback integral control (Extension) maintains this force. 
