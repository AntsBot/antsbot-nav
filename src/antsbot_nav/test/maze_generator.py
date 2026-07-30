import random
from random import Random
from typing import Optional
import numpy as np

def generate_maze(
    size: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a perfect maze using recursive backtracking.

    Parameters
    ----------
    size
        Maze size as (rows, cols). Both values must be odd.
    start
        Entrance cell (row, col).
    end
        Exit cell (row, col).
    seed
        Random seed.

    Returns
    -------
    np.ndarray
        Maze where
        - 0 = free
        - 1 = wall
    """
    rows, cols = size

    if rows % 2 == 0 or cols % 2 == 0:
        raise ValueError("Maze size must be odd.")

    rng = Random(seed)

    maze = [[1] * cols for _ in range(rows)]

    directions = [
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
    ]

    def dfs(r: int, c: int) -> None:
        maze[r][c] = 0

        dirs = directions[:]
        rng.shuffle(dirs)

        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc

            if not (1 <= nr < rows - 1 and 1 <= nc < cols - 1):
                continue

            if maze[nr][nc] == 0:
                continue

            maze[r + dr // 2][c + dc // 2] = 0
            dfs(nr, nc)

    dfs(1, 1)

    maze[start[0]][start[1]] = 0
    maze[end[0]][end[1]] = 0

    return np.array(maze)

