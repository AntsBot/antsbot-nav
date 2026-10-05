import numpy as np

class FireOriginPosition:
    """A single fire origin position on the map.
    Attributes:
        position (tuple[int, int]): The (x, y) coordinates of the fire origin.
        X (int): The x-coordinate of the fire origin.
        Y (int): The y-coordinate of the fire origin.
    """
    def __init__(self, position: tuple[int, int]):
        self.position = position
        self.X, self.Y = position
        
    def validate_out_of_bounds(self, grid_shape: tuple[int, int]) -> bool:
        """Check if the fire origin position is out of bounds of the grid."""
        return not (0 <= self.X < grid_shape[0] and 0 <= self.Y < grid_shape[1])
