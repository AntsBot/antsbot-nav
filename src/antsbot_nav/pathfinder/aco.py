from dataclasses import dataclass, field
import numpy as np

from . import ComputeBackend

Node = tuple[int, int]
Edge = tuple[Node, Node]

@dataclass
class ACOBase:
    num_ants: int = 50
    max_iterations: int = 100
    alpha: float = 1.0
    beta: float = 7.0
    evaporation_rate: float = 0.3
    initial_pheromone: float = 8.0
    pheromone_constant: float = 1.0
    backend: ComputeBackend = ComputeBackend.NUMPY

    best_path: list | None = field(default=None, init=False)
    best_cost: float = field(default=float("inf"), init=False)
    
    def _initialize_pheromones(self, shape: tuple[int, int]) -> None:
            """Initializes the pheromone levels for the given problem shape."""
            self._pheromones = np.full(shape, self.initial_pheromone)
        
    @staticmethod
    def _roulette_select(probabilities: np.ndarray) -> int:
        """Selects an index via roulette wheel selection.
    
        Args:
             probabilities (np.ndarray): Array of probabilities for each index.
    
        Returns:
            int: Selected index based on the probabilities.
        """
        cumulative = np.cumsum(probabilities)
        idx = np.searchsorted(cumulative, np.random.rand())
        return int(min(idx, len(probabilities) - 1))
    
    @staticmethod
    def _edge_key(from_node: Node, to_node: Node) -> Edge:
        """Return the canonical key for an undirected grid edge."""
        return (from_node, to_node) if from_node <= to_node else (to_node, from_node)