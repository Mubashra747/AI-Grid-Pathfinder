import pygame
import sys
import random
from collections import deque

# Initialize pygame
pygame.init()

WIDTH = 600
ROWS = 20
CELL_SIZE = WIDTH // ROWS

WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("GOOD PERFORMANCE TIME APP")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.color = WHITE
        self.parent = None

    def draw(self):
        pygame.draw.rect(
            WIN,
            self.color,
            (self.col * CELL_SIZE, self.row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

def create_grid():
    grid = []
    for i in range(ROWS):
        grid.append([])
        for j in range(ROWS):
            grid[i].append(Node(i, j))
    return grid

def draw_grid_lines():
    for i in range(ROWS):
        pygame.draw.line(WIN, BLACK, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE))
        pygame.draw.line(WIN, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, WIDTH))

def draw(grid):
    WIN.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw()
    draw_grid_lines()
    pygame.display.update()

# 🔥 Reconstruct Final Path
def reconstruct_path(start, end):
    current = end
    while current != start:
        current = current.parent
        if current != start:
            current.color = PURPLE
        draw(grid)
        pygame.time.delay(40)


# 🔥 BFS Algorithm with Animation + Dynamic Obstacles
def bfs(grid, start, end):
    queue = deque()
    queue.append(start)
    visited = set()

    directions = [
        (-1, 0), (0, 1), (1, 0), (1, 1),
        (0, -1), (-1, -1), (-1, 1), (1, -1)
    ]

    while queue:

        pygame.time.delay(50)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 🔥 Dynamic obstacle spawn
        if random.random() < 0.02:
            r = random.randint(0, ROWS - 1)
            c = random.randint(0, ROWS - 1)
            node = grid[r][c]
            if node.color == WHITE:
                node.color = BLACK

        current = queue.popleft()

        if current == end:
            reconstruct_path(start, end)
            return True

        visited.add((current.row, current.col))

        for dr, dc in directions:
            r = current.row + dr
            c = current.col + dc

            if 0 <= r < ROWS and 0 <= c < ROWS:
                neighbor = grid[r][c]

                if (r, c) not in visited and neighbor.color not in (BLACK, GREEN):
                    neighbor.parent = current
                    neighbor.color = YELLOW
                    queue.append(neighbor)

        if current != start:
            current.color = RED

        draw(grid)

    return False


# 🔥 MAIN GAME LOOP (This keeps window alive)
def main():
    global grid
    grid = create_grid()

    start = grid[2][2]
    end = grid[15][15]

    start.color = GREEN
    end.color = BLUE

    running = True
    while running:
        draw(grid)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Press SPACE to start BFS
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bfs(grid, start, end)

    pygame.quit()

main()
