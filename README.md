# Efficient Path planning of USV through Sunderban delta using AMRA*
Unmanned Surface Vehicles (USVs) are use to record oceanographic data. Traditional path planning used -fixed resolution and heuristic. But time constraints demand quick solutions.
Here Multiple heuristics are used to share search lists, Multiple resolutions are used to expand over varying step size, Anytime component to balance between time to reach goal and suboptimality of solution.
![image](https://user-images.githubusercontent.com/105581851/235601325-e4cfd1fa-01de-43aa-afb9-a1b769e428f5.png)

# Anytime multi-resolution multi-heuristic A*
Each (anytime) iteration returns a solution based on the (w1,w2) suboptimality bound. The Subsequent iterations are run by sequentially decrementing (w1,w2). State expansion is based on the heuristic.
Successors and actions taken are based on resolution. Run AMRA* starting from w1=500, w2=500, decremented appropriately till w1=1.1, w2=1.1;
![Sunderbans_output_w=1 1](https://user-images.githubusercontent.com/105581851/235628408-83ab3360-b30b-4c8f-a08b-feaf438f15d6.png)

![Sunderbans_output_extended_w=1 1](https://user-images.githubusercontent.com/105581851/235628590-2bc79d31-185f-4e38-9dd3-0591dde2214f.png)


Details in the attached Report..



