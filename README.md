# Efficient Path planning of USV through Sunderban delta using AMRA*
Unmanned Surface Vehicles (USVs) are use to record oceanographic data. Traditional path planning used -fixed resolution and heuristic. But time constraints demand quick solutions.
Here Multiple heuristics are used to share search lists, Multiple resolutions are used to expand over varying step size, Anytime component to balance between time to reach goal and suboptimality of solution.
![image](https://user-images.githubusercontent.com/105581851/235601325-e4cfd1fa-01de-43aa-afb9-a1b769e428f5.png)

# Anytime multi-resolution multi-heuristic A*
Each (anytime) iteration returns a solution based on the (w1,w2) suboptimality bound. The Subsequent iterations are run by sequentially decrementing (w1,w2). State expansion is based on the heuristic.
Successors and actions taken are based on resolution.

Details in the attached Report..



