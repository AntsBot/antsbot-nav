from dataclasses import dataclass, field, InitVar
from typing import Literal, ClassVar
from enum import Enum
from numpy.typing import ArrayLike
import numpy as np

from ..map.danger_map import DangerMap

__all__ = ["ComputeBackend", "ProblemType", "GridProblem", "AACOGridProblem", "TSPProblem"]

class ComputeBackend(Enum):
    NUMPY = "numpy"
    RUST = "rust"

class ProblemType(Enum):
    GRID = "grid"
    TSP = "tsp"

@dataclass
class GridProblem:
    _CARDINAL_DIRS: ClassVar[tuple[tuple[int, int], ...]] = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
    )
    _DIAGONAL_DIRS: ClassVar[tuple[tuple[int, int], ...]] = (
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )

    grid: ArrayLike  # type: ignore
    start: tuple[int, int]
    goal: tuple[int, int]
    connectivity: Literal[4, 8] = field(default=4, kw_only=True)
    allow_corner_cutting: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        self.grid: np.ndarray = np.asarray(self.grid)
        if self.grid.ndim != 2:
            raise ValueError("'grid' must be a 2D array.")
        if self.connectivity not in (4, 8):
            raise ValueError("'connectivity' must be either 4 or 8.")
        for name, point in (("start", self.start), ("goal", self.goal)):
            if not self._in_bounds(point):
                raise ValueError(f"'{name}' {point} is out of grid bounds.")
            if self.grid[point] != 0:
                raise ValueError(f"'{name}' {point} is on an obstacle.")

    def _in_bounds(self, point: tuple[int, int]) -> bool:
        row, col = point
        rows, cols = self.grid.shape
        return 0 <= row < rows and 0 <= col < cols

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def is_free(self, point: tuple[int, int]) -> bool:
        return self.grid[point] == 0

    def get_neighbors(self, point: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = point
        directions = self._CARDINAL_DIRS
        if self.connectivity == 8:
            directions += self._DIAGONAL_DIRS
        neighbors: list[tuple[int, int]] = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < self.shape[0] and 0 <= nc < self.shape[1]):
                continue
            if self.grid[nr, nc] != 0:
                continue
            if (
                self.connectivity == 8
                and not self.allow_corner_cutting
                and abs(dr) == 1
                and abs(dc) == 1
            ):
                if self.grid[row + dr, col] != 0 or self.grid[row, col + dc] != 0:
                    continue
            neighbors.append((nr, nc))
        return neighbors

@dataclass
class AACOGridProblem(GridProblem):
    danger_map: DangerMap = field(kw_only=True)

@dataclass
class TSPProblem:
    coordinates: np.ndarray
    distance_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.coordinates = self._validate_coordinates(self.coordinates)
        self.distance_matrix = self._coordinates_to_distance_matrix(self.coordinates)

    @staticmethod
    def _validate_coordinates(coordinates: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("'coordinates' must have shape (n, 2).")
        return coordinates

    @staticmethod
    def _coordinates_to_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
        diff = coordinates[:, None, :] - coordinates[None, :, :]
        return np.linalg.norm(diff, axis=2)

    @property
    def num_cities(self) -> int:
        return len(self.coordinates)
