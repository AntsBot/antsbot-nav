from antsbot_nav.test import generate_maze
from antsbot_nav.pathfinder import GridACO,AStar, GridProblem

maze = generate_maze((11, 11), (1, 1), (9, 9), seed=42)
maze2 = generate_maze((111, 111), (1, 1), (109, 109), seed=42)

problem = GridProblem(maze, (1, 1), (9, 9))
problem2 = GridProblem(maze2, (1, 1), (109, 109))

aco = GridACO(num_ants=100, max_iterations=250, alpha=1.0, beta=7.0, evaporation_rate=0.3, initial_pheromone=8.0, pheromone_constant=1.0)
aco_deep = GridACO(num_ants=1000, max_iterations=750, alpha=1.0, beta=2.0, evaporation_rate=0.1, initial_pheromone=8.0, pheromone_constant=1.0)
astar = AStar()

aco.load(problem2)
aco_deep.load(problem2)
astar.load(problem2)


best_path, best_cost = aco_deep.run(progress=True)
best_path_astar, best_cost_astar = astar.run()

print("Best path:", best_path)
print("Best cost:", best_cost)
print("Best path A*:", best_path_astar)
print("Best cost A*:", best_cost_astar)