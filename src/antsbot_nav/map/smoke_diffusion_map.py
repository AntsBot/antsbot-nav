from dataclasses import dataclass, field
import numpy as np

@dataclass
class SmokeDiffusionMap:
    """
    Represents a smoke diffusion map for pathfinding algorithms.
    
    Attributes:
        base_map (np.ndarray): 2D array representing the base levels in the environment.
        smoke_map (np.ndarray): 2D array representing the smoke levels in the environment.
    """
    

    base_map: np.ndarray
    smoke_map: np.ndarray = field(init=False)

    def __post_init__(self):
        self.smoke_map = np.zeros_like(self.base_map, dtype=np.float32)
        
    