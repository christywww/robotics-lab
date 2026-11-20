NOTE:  Path.py and Control.py both require pressing enter before displaying the path or running the simulation!

### Inverse Geometry

We use scipy's `least_squares` solver with error (effMcube for both hands) plus postural bias as residual and joint limits as bounds. Results are rejected if in collision or if cube and robot distance is less than 1cm from obstacle, with solver retry from nearby starting configuration. A slower SLSQP solver using collision distance as constraint serves as fallback after 10 failed attempts.

### RRT(-connect)

At each step in `new_conf` and during `rand_conf`, `computegrasppose` is called on the cube position to generate joint configurations grasping the cube.

To speed up computepath, we implemented RRT-connect so the path is found in fewer iterations using the faster `least_squares` solver for `computegrasppose`. While `least_squares` sometimes fails to find valid q configurations, it is fine to reject it because RRT will move on to the next node and the path continues.

### Trajectory

The trajectory is a chain of Bézier curves joined so that position, velocity and acceleration are continuous at joins. Bézier curves are computed by solving a quadratic program using quadprog. The cost is a least squares fit to the path plus acceleration penalty. Curves are constrained to joint limits and trajectory starts and ends at rest.

### Torque Control

A torque PID controller based on joint error plus feedforward acceleration generates a desired acceleration, converted to joint torques via RNEA (Pinocchio). A squeeze torque ensures grip.

### Additional work.

**K-d Tree**
RRTTree.py implements the K-d Tree structure and improved nearest_vertex search for RRT, which scales better with more nodes (O(logn) on average) compared to an O(n) brute-force search.


**Force-feedback integral control**
The torque controller has been improved to maintain the force from end effectors on the cube by measuring the error between signal force and actual cube force experienced.
