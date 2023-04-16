# Force-Directed-layout
impelementation of force directed layout with python （numpy and pytorch）  
## Content 
The main purpose of this experiment is to simulate the force-directed layout algorithm. Therefore, only the following simple mechanical simulations will be carried out:
- Repulsive forces between particles
- Treating edges as springs, calculating the spring force based on distance and the ideal length of the spring.
## Algorithm Steps
1. Import edge information from a file.
2. Generate position information for 499 nodes.
3. Calculate the repulsive forces acting on each point.
4. Calculate the spring forces acting on each point.
5. Calculate the resultant force on each point based on (3) and (4), and update its coordinates.
6， Repeat steps (3) to (5) until convergence or the specified target is reached.   
### 1. Repulsion Computation     
Two points $A(x_A,y_A)$ and $B(x_B,y_B)$
#### 1.1 Calculate the distance   
$$distance = \sqrt{(x_A - x_B)^2 + (y_A - y_B)^2}$$
#### 1.2 Calculated the total repulsive force
- For nodes without mass   
$$F_{repulsion} = \frac{k_r}{distance^2}$$
- For nodes with mass
$$F_{repulsion} = \frac{k_r * m_A * m_B * y}{distance^2}$$
#### 1.3 Calculate the component forces on the x and y axis
$$\Delta x, \Delta y = x_A - x_B, y_A - y_B$$

$$𝐹_{𝑟𝑒𝑝𝑢𝑙𝑠𝑖𝑜𝑛_𝑋}=𝐹_{𝑟𝑒𝑝𝑢𝑙𝑠𝑖𝑜𝑛}∗Δ𝑥/𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒$$

$$𝐹_{𝑟𝑒𝑝𝑢𝑙𝑠𝑖𝑜𝑛_Y}=𝐹_{𝑟𝑒𝑝𝑢𝑙𝑠𝑖𝑜𝑛}∗Δy/𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒$$
#### 1.4 Update the repulsive forces experienced by A and B respectively
$$repulsion_{A_X} += F_{repulsion_X}$$ 

$$repulsion_{A_Y} += F_{repulsion_Y}$$

$$repulsion_{B_X} -= F_{repulsion_X}$$

$$repulsion_{B_Y} -= F_{repulsion_Y}$$
### 2. Spring force computation
$$F_{spring}=k_s\cdot(distance-dis_{ideal})$$
where $𝑘_𝑠$ is a coefficient set by the user to represent the spring's elasticity,
$𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒$ represents the distance between two points, which is the current length of the spring,
$𝑑𝑖𝑠_𝑖𝑑𝑒𝑎𝑙$ represents the ideal length of the spring, which is the ideal distance between two points.

Then update the spring forces experienced by A and B respectively.
### 3. Calculate the resultant force on each point
For each point $N(x_N,y_N)$
$$\Delta x_N,\Delta y_N=repulsion_N+spring_N$$
  ### 4. Update coordinates
$$𝑥_𝑁^′,𝑦_𝑁^′=𝑥_𝑁+Δ𝑥_𝑁,y_N+Δ𝑦_𝑁$$
## Running the code
- File structure
```
|   n100-withmass_fps_120.mp4   #demo video
|   n499-nomass_fps_120.mp4  #demo video
|   ForceDirected.py  #basic implementation
|   NumpyForceDirected.py  #implementation with numpy
|   PytorchNumpyForceDirected.py  #implementation with pytorch and numpy
|   writeVideo.py  #transfer imgs to video
+---imgs  saving results
\---saved_data #initial files
        edges_epoch_latest.txt  
        network.txt
        nodes_epoch_latest.txt
```

- For help
```
python [xxx.py] -h
```
- Quick startup  
Run and make dir ```imgs/test```, img of each epoch is saved in ```imgs/test/imgs/```
```
python ForceDirected.py --load_edge --node_num 499 --dir test
```
Get demo video
```
python writeVideo.py --fps 200 --dir test --name test_video
```
- recommendation  
enable ```--notsave --notdraw``` to save running time

