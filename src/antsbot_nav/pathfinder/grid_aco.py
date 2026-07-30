from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
from .types import ComputeBackend, GridProblem, ProblemType, TSPProblem
from .aco import ACOBase, Node, Edge

@dataclass
class GridACO(ACOBase):
    """Ant Colony Optimization (ACO) algorithm for pathfinding.

    This implementation is specifically designed for grid-based pathfinding problems.

    Attributes:
        num_ants (int): Number of ants in the colony.
        max_iterations (int): Maximum number of iterations to run the algorithm.
        alpha (float): Influence of pheromone trails on path selection.
        beta (float): Influence of heuristic information on path selection.
        evaporation_rate (float): Rate at which pheromone trails evaporate.
        initial_pheromone (float): Initial pheromone level on paths.
        pheromone_constant (float): Constant used to update pheromone levels.
        backend (Backend): Backend to use for computations.
    """
    # To avoid infinite loops, limit ant steps.
    _max_steps: int = field(default=0, init=False, repr=False)



    def _initialize_pheromones(self, shape: tuple[int, int]) -> None:
        """Initialize one pheromone value for every traversable grid edge."""
        self._pheromones: dict[Edge, float] = {}
        for row in range(shape[0]):
            for col in range(shape[1]):
                node = (row, col)
                if not self.problem.is_free(node):
                    continue
                for neighbor in self.problem.get_neighbors(node):
                    self._pheromones[self._edge_key(node, neighbor)] = self.initial_pheromone

    def load(self, problem: GridProblem) -> "GridACO":
        """Attach a problem instance to the solver and derive its shape.

        Args:
            problem: A `GridProblem` to solve.

        Returns:
            GridACO: self, to allow chained calls (e.g. `GridACO().load(problem).run()`).
        """
        if isinstance(problem, GridProblem):
            self.problem_type = ProblemType.GRID
            self.problem = problem
            pheromone_shape = problem.shape
        else:
            raise TypeError(f"Unsupported problem type: {type(problem).__name__}")

        self._initialize_pheromones(pheromone_shape)
        # Set Ant steps limit to 2x grid size
        self._max_steps = 2 * pheromone_shape[0] * pheromone_shape[1]
        return self


    def _update_pheromones(self, paths: list[list[Node]], costs: list[float]) -> None:
        """Update the pheromone levels based on the paths taken by the ants.

        Args:
            paths (list[list[tuple[int, int]]]): List of paths taken by each ant.
            costs (list[float]): List of costs associated with each path.
        """
        for edge in self._pheromones:
            self._pheromones[edge] *= 1 - self.evaporation_rate

        for path, cost in zip(paths, costs):
            if cost <= 0:
                continue
            pheromone_deposit = self.pheromone_constant / cost
            for from_node, to_node in zip(path, path[1:]):
                edge = self._edge_key(from_node, to_node)
                self._pheromones[edge] += pheromone_deposit

    def _compute_heuristic(self, from_node: tuple[int, int], to_node: tuple[int, int]) -> float:
        dr = abs(from_node[0] - to_node[0])
        dc = abs(from_node[1] - to_node[1])
        if self.problem.connectivity == 8:
            distance = max(dr, dc)
        else:
            distance = dr + dc
        if distance == 0:
            return 1.0
        return 1.0 / distance

    def _get_unvisited_neighbors(
        self, current_node: tuple[int, int], visited: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Get neighbors of a node that haven't already been visited on this path.
 
        Args:
            current_node (tuple[int, int]): The node to get neighbors of.
            visited (set[tuple[int, int]]): Nodes already visited on the current path.
 
        Returns:
            list[tuple[int, int]]: Neighbors not yet visited, in the order
            returned by the underlying problem.
        """
        return [n for n in self.problem.get_neighbors(current_node) if n not in visited]
 
    def _transition_probabilities(
        self, current_node: Node, neighbors: list[Node], end: Node
    ) -> np.ndarray:
        """Compute the normalized transition probability for each candidate neighbor.
 
        Combines pheromone level and goal-distance heuristic per the standard
        ACO transition rule:
 
            p_i = (tau_(current, i)^alpha * eta_i^beta)
                  / sum_k(tau_(current, k)^alpha * eta_k^beta)
 
        Args:
            current_node: The node from which the ant will move.
            neighbors (list[tuple[int, int]]): Candidate next nodes.
            end (tuple[int, int]): The path's target node, used to score
                each candidate's heuristic desirability.
 
        Returns:
            np.ndarray: Probabilities aligned with `neighbors`, summing to 1.
            Falls back to a uniform distribution if the weighted scores are
            degenerate (all zero, or non-finite from an `inf` heuristic).
        """
        weights = np.array(
            [
                (self._pheromones[self._edge_key(current_node, n)] ** self.alpha)
                * (self._compute_heuristic(n, end) ** self.beta)
                for n in neighbors
            ]
        )
 
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            if np.isfinite(total):
                return np.full(len(neighbors), 1.0 / len(neighbors))
            is_inf = ~np.isfinite(weights)
            return is_inf.astype(float) / is_inf.sum()
 
        return weights / total

    def _build_path(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        """Build a path from the start node to the end node using the ACO algorithm.
 
        Args:
            start (tuple[int, int]): The starting node coordinates.
            end (tuple[int, int]): The target node coordinates.
 
        Returns:
            list[tuple[int, int]] | None: The built path from start to end,
            or None if the ant got stuck (dead end) or exceeded the step
            budget before reaching `end`.
        """
        path = [start]
        visited = {start}
        current_node = start
        steps = 0
 
        while current_node != end:
            if steps >= self._max_steps:
                return None  # exceeded budget, likely unreachable or stuck
 
            neighbors = self._get_unvisited_neighbors(current_node, visited)
            if not neighbors:
                return None  # dead end
 
            probabilities = self._transition_probabilities(current_node, neighbors, end)
            next_node_index = self._roulette_select(probabilities)
            next_node = neighbors[next_node_index]
 
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
            steps += 1
 
        return path
 
    @staticmethod
    def _path_cost(path: list[tuple[int, int]]) -> float:
        """Cost of a path, defined as its number of steps (edges).

        Args:
            path (list[tuple[int, int]]): The path to evaluate.

        Returns:
            float: Number of edges traversed. A single-node path costs 0.
        """
        return float(len(path) - 1)

    def run(self, progress: bool = False) -> tuple[list, float]:
        """Run the ACO algorithm to find a path from start to end.

        Must be called after `load()`. Iterates for `max_iterations` rounds,
        each round sending `num_ants` ants out to build a path, then updating
        pheromones based on the paths found. Tracks the best (lowest-cost)
        path seen across all iterations.
        
        Args:
            progress (bool): If True, display a progress bar for iterations.

        Returns:
            tuple[list, float]: The best path found and its cost. If no ant
            ever reached `end`, returns `(None, float('inf'))`.
        """
        if not hasattr(self, "problem"):
            raise RuntimeError("No problem loaded — call `load()` before `run()`.")
        
        if progress:
            iterator = tqdm(range(self.max_iterations), desc="ACO Iterations", position=0)
            
        else:
            iterator = range(self.max_iterations)
            ants_iterator = range(self.num_ants)

        start, end = self.problem.start, self.problem.goal

        for iter in iterator:
            paths = []
            costs = []
            
            if progress:
                ants_iterator = tqdm(range(self.num_ants), desc=f"Ants Exploring...", position=1, leave=False)
            else:
                ants_iterator = range(self.num_ants)

            for ant in ants_iterator :
                path = self._build_path(start, end)
                if path is None:
                    # ant failed to reach the target this round
                    continue  

                cost = self._path_cost(path)
                paths.append(path)
                costs.append(cost)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = path

            if paths:
                self._update_pheromones(paths, costs)

        return self.best_path, self.best_cost # pyright: ignore[reportReturnType]
