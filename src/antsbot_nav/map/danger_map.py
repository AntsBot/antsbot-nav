import numpy as np

class DangerMap:
    """
    Represents a danger map for pathfinding algorithms.
    
    Attributes:
        base_map (np.ndarray): 2D array representing the base levels in the environment.
    """

    def __init__(self, base_map: np.ndarray):
        self.base_map = base_map
        self.smoke_map = np.zeros_like(base_map, dtype=np.float32)
        
        
        