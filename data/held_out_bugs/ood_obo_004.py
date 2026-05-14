def get_neighbors(grid: list[list[int]], r: int, c: int) -> list[int]:
    neighbors = []
    for dr in range(-1, 1):
        for dc in range(-1, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbors.append(grid[nr][nc])
    return neighbors
