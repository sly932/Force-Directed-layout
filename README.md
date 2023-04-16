# Force-Directed-layout
impelementation of force directed layout with python （numpy and pytorch）  
### Content 
The main purpose of this experiment is to simulate the force-directed layout algorithm. Therefore, only the following simple mechanical simulations will be carried out:
- Repulsive forces between particles
- Treating edges as springs, calculating the spring force based on distance and the ideal length of the spring.
### Algorithm Steps
- Import edge information from a file.
- Generate position information for 499 nodes.
- Calculate the repulsive forces acting on each point.
- Calculate the spring forces acting on each point.
- Calculate the resultant force on each point based on (3) and (4), and update its coordinates.
- Repeat steps (3) to (5) until convergence or the specified target is reached.   
### Repulsion Computation     
Two points $A(x_A,y_A)$ and $B(x_B,y_B)$
#### Calculate the distance   
$$distance = \sqrt{(x_A - x_B)^2 + (y_A - y_B)^2}$$
#### Calculated the total repulsive force
- For nodes without mass   
$$F_{repulsion} = \frac{k_r}{distance^2}$$
- For nodes with mass
$$F_{repulsion} = \frac{k_r * m_A * m_B * y}{distance^2}$$
#### Calculate the component forces on the x and y axis
$$\Delta x, \Delta y = x_A - x_B, y_A - y_B$$  
 𝐹_{𝑟𝑒𝑝𝑢𝑙𝑠𝑖𝑜𝑛}_𝑋=𝐹_{𝑟𝑒𝑝𝑢𝑙𝑠𝑖𝑜𝑛}∗Δ𝑥/𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒
 𝐹_{𝑟𝑒𝑝𝑢𝑙𝑠𝑖𝑜𝑛}_Y=𝐹_{𝑟𝑒𝑝𝑢𝑙𝑠𝑖𝑜𝑛}∗Δy/𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒     
$$   
#### Update the repulsive forces experienced by A and B respectively
$$repulsion_{A_X} += F_{repulsion_X}$$ 
$$repulsion_{A_Y} += F_{repulsion_Y}$$



