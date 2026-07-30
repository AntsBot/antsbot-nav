from .types import GridProblem

class AStar:
    """Find shortest paths through a :class:`GridProblem` using A*.

    Every valid grid move has a cost of one.  Consequently, Manhattan
    distance is an admissible heuristic for four-connected grids and
    Chebyshev distance is admissible for eight-connected grids.
    """

    def load(self, problem: GridProblem) -> "AStar":
        """Load the problem into the A* algorithm.

        Args:
            problem (GridProblem): The grid problem to solve.
        """
        if not isinstance(problem, GridProblem):
            raise TypeError(f"Unsupported problem type: {type(problem).__name__}")
        self.problem = problem
        return self

    def _compute_heuristic(self, point: tuple[int, int]) -> int:
        """Return the admissible remaining-cost estimate for ``point``."""
        row_distance = abs(point[0] - self.problem.goal[0])
        col_distance = abs(point[1] - self.problem.goal[1])
        if self.problem.connectivity == 8:
            return max(row_distance, col_distance)
        return row_distance + col_distance

    @staticmethod
    def _rebuild_path(
        came_from: dict[tuple[int, int], tuple[int, int]],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Rebuild the path ending at ``goal`` from predecessor links."""
        path = [goal]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def run(self) -> tuple[list[tuple[int, int]] | None, float]:
        """Find an optimal path from the loaded problem's start to its goal.

        Returns:
            The path (including start and goal) and its number of moves.  If
            the goal is unreachable, returns ``(None, float('inf'))``.

        Raises:
            RuntimeError: If called before :meth:`load`.
        """
        if not hasattr(self, "problem"):
            raise RuntimeError("No problem loaded — call `load()` before `run()`.")

        start = self.problem.start
        goal = self.problem.goal
        if start == goal:
            return [start], 0.0

        # Keep the nodes waiting to be explored in a plain list.  The node
        # with the smallest estimated total cost is selected each iteration.
        open_set = [start]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], int] = {start: 0}

        while open_set:
            current = min(
                open_set,
                key=lambda point: g_score[point] + self._compute_heuristic(point),
            )
            open_set.remove(current)
            current_cost = g_score[current]

            if current == goal:
                return self._rebuild_path(came_from, goal), float(current_cost)

            for neighbor in self.problem.get_neighbors(current):
                tentative_cost = current_cost + 1
                if tentative_cost >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                if neighbor not in open_set:
                    open_set.append(neighbor)

        return None, float("inf")
