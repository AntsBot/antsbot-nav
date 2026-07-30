from dataclasses import dataclass, field
import numpy as np

from .types import AACOGridProblem, ProblemType
from .grid_aco import GridACO


@dataclass
class AACO(GridACO):
    """ Adaptive Ant Colony Optimization (AACO) algorithm for pathfinding. """
    
    def load(self, problem: AACOGridProblem) -> "AACO":
        """Attach a problem instance to the solver and derive its shape.

        Args:
            problem: A `GridProblem` to solve.

        Returns:
            AACO: self, to allow chained calls (e.g. `AACO().load(problem).run()`).
        """
        if isinstance(problem, AACOGridProblem):
            self.problem_type = ProblemType.GRID
            self.problem = problem
            pheromone_shape = problem.shape
        else:
            raise TypeError(f"Unsupported problem type: {type(problem).__name__}")

        self._initialize_pheromones(pheromone_shape)
        # Set Ant steps limit to 2x grid size
        self._max_steps = 2 * pheromone_shape[0] * pheromone_shape[1]
        return self
    
    def _compute_heuristic(self, from_node: tuple[int, int], to_node: tuple[int, int]) -> float:
        dr = abs(from_node[0] - to_node[0])
        dc = abs(from_node[1] - to_node[1])
        if self.problem.connectivity == 8:
            distance = max(dr, dc)
        else:
            distance = dr + dc
        if distance == 0:
            return 1.0
        
        return 1.0 / (distance + self.problem.danger_map.smoke_map[to_node])