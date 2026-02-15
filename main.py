import pygame
import sys
import random
import heapq
from collections import deque

pygame.init()

# ===============================
# WINDOW SETTINGS
# ===============================
GRID_SIZE = 600
SIDE_PANEL = 220
WIDTH = GRID_SIZE + SIDE_PANEL
HEIGHT = 600
ROWS = 20
CELL_SIZE = GRID_SIZE // ROWS

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GOOD PERFORMANCE TIME APP")

# ===============================
# COLORS
# ===============================
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)

# ===============================
# STATISTICS TRACKER
# ===============================
class StatsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.nodes_explored = 0
        self.start_time = 0
        self.end_time = 0
        self.algorithm_name = ""
        self.show_stats = False
    
    def start(self, algo_name):
        self.reset()
        self.algorithm_name = algo_name
        self.start_time = pygame.time.get_ticks()
        self.show_stats = True
    
    def stop(self):
        self.end_time = pygame.time.get_ticks()
    
    def add_node(self):
        self.nodes_explored += 1
    
    def get_time_ms(self):
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0
    
    def draw(self, surface):
        if self.show_stats and self.algorithm_name:
            font = pygame.font.SysFont(None, 24)
            
            # Draw background with border
            stats_rect = pygame.Rect(GRID_SIZE + 10, HEIGHT - 80, 200, 70)
            pygame.draw.rect(surface, GRAY, stats_rect)
            pygame.draw.rect(surface, BLACK, stats_rect, 2)
            
            # Draw stats text
            algo_text = font.render(f"Algorithm: {self.algorithm_name}", True, BLACK)
            nodes_text = font.render(f"Nodes: {self.nodes_explored}", True, BLACK)
            time_text = font.render(f"Time: {self.get_time_ms()}ms", True, BLACK)
            
            surface.blit(algo_text, (GRID_SIZE + 20, HEIGHT - 70))
            surface.blit(nodes_text, (GRID_SIZE + 20, HEIGHT - 45))
            surface.blit(time_text, (GRID_SIZE + 20, HEIGHT - 20))

# ===============================
# NO PATH FOUND FUNCTION
# ===============================
def show_no_path_message():
    """Display message when no path exists"""
    font = pygame.font.SysFont(None, 48)
    text = font.render("NO PATH FOUND!", True, RED)
    text_rect = text.get_rect(center=(GRID_SIZE//2, HEIGHT//2 - 50))
    
    # Draw semi-transparent overlay
    s = pygame.Surface((GRID_SIZE, HEIGHT))
    s.set_alpha(180)
    s.fill(WHITE)
    WIN.blit(s, (0, 0))
    
    # Draw message
    WIN.blit(text, text_rect)
    pygame.display.update()
    pygame.time.delay(1500)  # Show for 1.5 seconds

# ===============================
# NODE CLASS
# ===============================
class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.color = WHITE
        self.parent = None

    def draw(self, surface):
        pygame.draw.rect(
            surface,
            self.color,
            (self.col * CELL_SIZE, self.row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

# ===============================
# GRID FUNCTIONS
# ===============================
def create_grid():
    return [[Node(i, j) for j in range(ROWS)] for i in range(ROWS)]

def draw_grid_lines(surface):
    for i in range(ROWS):
        pygame.draw.line(surface, BLACK, (0, i * CELL_SIZE), (GRID_SIZE, i * CELL_SIZE))
        pygame.draw.line(surface, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE))

def draw_buttons(surface):
    font = pygame.font.SysFont(None, 24)
    button_width = 180
    button_height = 40
    margin = 15

    start_x = GRID_SIZE + 20
    start_y = 20

    names = ["BFS", "DFS", "UCS", "DLS", "IDDFS", "BIDIRECTIONAL", "RESET"]
    rects = {}

    for i, name in enumerate(names):
        y = start_y + i * (button_height + margin)
        rect = pygame.Rect(start_x, y, button_width, button_height)
        pygame.draw.rect(surface, GRAY, rect)
        pygame.draw.rect(surface, BLACK, rect, 2)

        text = font.render(name, True, BLACK)
        surface.blit(text, (start_x + 30, y + 10))
        rects[name] = rect

    return rects

def draw(grid, surface, stats):
    surface.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(surface)

    draw_grid_lines(surface)
    button_rects = draw_buttons(surface)
    stats.draw(surface)  # Draw stats last to ensure visibility

    pygame.display.update()
    return button_rects

# ===============================
# PATH RECONSTRUCTION
# ===============================
def reconstruct_path(start, end, grid, surface, stats):
    current = end
    path_nodes = []
    
    # Collect path nodes first
    while current != start:
        current = current.parent
        if current != start:
            path_nodes.append(current)
    
    # Animate path
    for node in reversed(path_nodes):
        node.color = PURPLE
        draw(grid, surface, stats)
        pygame.time.delay(30)
    
    # Final pause to show complete path
    pygame.time.delay(500)

def reconstruct_bidirectional_path(start, meeting, end, parent_s, parent_e, grid, surface, stats):
    """Reconstruct path from both directions after bidirectional search meets"""
    
    # Build path from start to meeting point
    path_to_meeting = []
    current = meeting
    while current != start:
        path_to_meeting.append(current)
        current = parent_s.get(current)
        if current is None:
            break
    
    # Build path from meeting to end
    path_to_end = []
    current = meeting
    while current != end:
        current = parent_e.get(current)
        if current is None:
            break
        path_to_end.append(current)
    
    # Color the complete path (excluding start and end)
    for node in reversed(path_to_meeting):
        if node != start and node != meeting:
            node.color = PURPLE
        draw(grid, surface, stats)
        pygame.time.delay(30)
    
    for node in path_to_end:
        if node != end and node != meeting:
            node.color = PURPLE
        draw(grid, surface, stats)
        pygame.time.delay(30)
    
    # Final pause to show complete path
    pygame.time.delay(1000)

# ===============================
# COMMON DIRECTIONS
# ===============================
directions = [
    (-1, 0), (0, 1), (1, 0), (1, 1),
    (0, -1), (-1, -1), (-1, 1), (1, -1)
]

# Create global stats tracker
stats = StatsTracker()

# ===============================
# BFS (with stats)
# ===============================
def bfs(grid, start, end, surface):
    stats.start("BFS")
    queue = deque([start])
    visited = set()

    while queue:
        pygame.time.delay(30)
        current = queue.popleft()
        stats.add_node()

        if current == end:
            stats.stop()
            reconstruct_path(start, end, grid, surface, stats)
            return True

        visited.add((current.row, current.col))

        for dr, dc in directions:
            r, c = current.row + dr, current.col + dc
            if 0 <= r < ROWS and 0 <= c < ROWS:
                neighbor = grid[r][c]
                if (r, c) not in visited and neighbor.color != GREEN:  # Only avoid start
                    neighbor.parent = current
                    if neighbor != end:
                        neighbor.color = YELLOW
                    queue.append(neighbor)

        if current != start and current != end:
            current.color = RED

        draw(grid, surface, stats)
    
    stats.stop()
    return False

# ===============================
# DFS (with stats)
# ===============================
def dfs(grid, start, end, surface):
    stats.start("DFS")
    stack = [start]
    visited = set()

    while stack:
        pygame.time.delay(30)
        current = stack.pop()
        stats.add_node()

        if current == end:
            stats.stop()
            reconstruct_path(start, end, grid, surface, stats)
            return True

        visited.add((current.row, current.col))

        for dr, dc in directions:
            r, c = current.row + dr, current.col + dc
            if 0 <= r < ROWS and 0 <= c < ROWS:
                neighbor = grid[r][c]
                if (r, c) not in visited and neighbor.color != GREEN:  # Only avoid start
                    neighbor.parent = current
                    if neighbor != end:
                        neighbor.color = YELLOW
                    stack.append(neighbor)

        if current != start and current != end:
            current.color = RED

        draw(grid, surface, stats)
    
    stats.stop()
    return False

# ===============================
# UCS - COMPLETELY FIXED VERSION
# ===============================
def ucs(grid, start, end, surface):
    stats.start("UCS")
    
    # Priority queue: (cost, id, node) - id prevents comparison errors
    counter = 0
    heap = []
    heapq.heappush(heap, (0, counter, start))
    counter += 1
    
    # Cost tracking
    costs = {(start.row, start.col): 0}
    visited = set()
    
    # Parent tracking for path reconstruction
    parent = {start: None}
    
    while heap:
        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        pygame.time.delay(30)
        
        # Get node with smallest cost
        cost, _, current = heapq.heappop(heap)
        
        # Skip if already visited
        if (current.row, current.col) in visited:
            continue
        
        # Mark as visited and count node
        visited.add((current.row, current.col))
        stats.add_node()
        
        # Color current node (except start/end)
        if current != start and current != end:
            current.color = RED
        
        # Check if reached goal
        if current == end:
            # Reconstruct path
            path = []
            node = end
            while node:
                path.append(node)
                node = parent.get(node)
            
            # Color path
            for node in reversed(path):
                if node != start and node != end:
                    node.color = PURPLE
                draw(grid, surface, stats)
                pygame.time.delay(30)
            
            stats.stop()
            pygame.time.delay(500)  # Pause to show result
            return True
        
        # Explore neighbors
        for dr, dc in directions:
            r, c = current.row + dr, current.col + dc
            
            # Check bounds
            if 0 <= r < ROWS and 0 <= c < ROWS:
                neighbor = grid[r][c]
                
                # Skip start node (only if it's not current path)
                if neighbor == start:
                    continue
                
                # Calculate cost
                if abs(dr) == 1 and abs(dc) == 1:
                    new_cost = cost + 1.4  # Diagonal cost
                else:
                    new_cost = cost + 1    # Straight cost
                
                # If better path found or not visited
                if (r, c) not in costs or new_cost < costs[(r, c)]:
                    costs[(r, c)] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(heap, (new_cost, counter, neighbor))
                    counter += 1
                    
                    # Color neighbor (if not end)
                    if neighbor != end and neighbor != start:
                        neighbor.color = YELLOW
        
        # Update display
        draw(grid, surface, stats)
    
    # No path found
    stats.stop()
    return False

# ===============================
# DLS (with stats)
# ===============================
def dls(grid, start, end, limit, surface):
    stats.start(f"DLS (limit={limit})")
    stack = [(start, 0)]
    visited = set()

    while stack:
        pygame.time.delay(30)
        current, depth = stack.pop()
        stats.add_node()

        if current == end:
            stats.stop()
            reconstruct_path(start, end, grid, surface, stats)
            return True

        if depth < limit:  # Changed from <= to < to prevent exceeding limit
            visited.add((current.row, current.col))
            for dr, dc in directions:
                r, c = current.row + dr, current.col + dc
                if 0 <= r < ROWS and 0 <= c < ROWS:
                    neighbor = grid[r][c]
                    if (r, c) not in visited and neighbor.color != GREEN:  # Only avoid start
                        neighbor.parent = current
                        if neighbor != end:
                            neighbor.color = YELLOW
                        stack.append((neighbor, depth + 1))

        if current != start and current != end:
            current.color = RED

        draw(grid, surface, stats)
    
    stats.stop()
    return False

# ===============================
# IDDFS (with stats)
# ===============================
def iddfs(grid, start, end, surface):
    stats.start("IDDFS")
    for depth in range(ROWS * ROWS):
        # Don't reset stats for each iteration
        if dls(grid, start, end, depth, surface):
            stats.stop()
            return True
        # Quick redraw to show iteration progress
        draw(grid, surface, stats)
    
    stats.stop()
    show_no_path_message()
    return False

# ===============================
# BIDIRECTIONAL (Fixed with stats)
# ===============================
def bidirectional(grid, start, end, surface):
    stats.start("BIDIRECTIONAL")
    
    # If start and end are the same
    if start == end:
        stats.stop()
        return True
    
    queue_s = deque([start])
    queue_e = deque([end])
    
    # Track visited nodes and their parents
    visited_s = {(start.row, start.col): start}
    visited_e = {(end.row, end.col): end}
    
    # Parent dictionaries for path reconstruction
    parent_s = {start: None}
    parent_e = {end: None}
    
    meeting_point = None
    
    while queue_s and queue_e and not meeting_point:
        pygame.time.delay(30)
        
        # Expand from start side
        if queue_s:
            current_s = queue_s.popleft()
            stats.add_node()
            
            for dr, dc in directions:
                r, c = current_s.row + dr, current_s.col + dc
                if 0 <= r < ROWS and 0 <= c < ROWS:
                    neighbor = grid[r][c]
                    
                    # Check if this node was visited from end side
                    if (r, c) in visited_e:
                        meeting_point = neighbor
                        parent_s[neighbor] = current_s
                        break
                    
                    # Add to start frontier if not visited and not start
                    if (r, c) not in visited_s and neighbor.color != GREEN:  # Only avoid start
                        visited_s[(r, c)] = neighbor
                        parent_s[neighbor] = current_s
                        if neighbor != end:
                            neighbor.color = YELLOW
                        queue_s.append(neighbor)
            
            if current_s != start and current_s != end:
                current_s.color = RED
            
            draw(grid, surface, stats)
        
        # Expand from end side
        if queue_e and not meeting_point:
            current_e = queue_e.popleft()
            stats.add_node()
            
            for dr, dc in directions:
                r, c = current_e.row + dr, current_e.col + dc
                if 0 <= r < ROWS and 0 <= c < ROWS:
                    neighbor = grid[r][c]
                    
                    # Check if this node was visited from start side
                    if (r, c) in visited_s:
                        meeting_point = neighbor
                        parent_e[neighbor] = current_e
                        break
                    
                    # Add to end frontier if not visited and not start
                    if (r, c) not in visited_e and neighbor.color != GREEN:  # Only avoid start
                        visited_e[(r, c)] = neighbor
                        parent_e[neighbor] = current_e
                        if neighbor != start:
                            neighbor.color = YELLOW
                        queue_e.append(neighbor)
            
            if current_e != end and current_e != start:
                current_e.color = RED
            
            draw(grid, surface, stats)
    
    # If meeting point found, reconstruct the path
    if meeting_point:
        # Clear the explored colors but keep start and end
        for row in grid:
            for node in row:
                if node.color not in (GREEN, BLUE):
                    node.color = WHITE
                if node == meeting_point:
                    node.color = YELLOW  # Highlight meeting point
        draw(grid, surface, stats)
        pygame.time.delay(500)
        
        # Reconstruct and show the complete path
        reconstruct_bidirectional_path(start, meeting_point, end, parent_s, parent_e, grid, surface, stats)
        stats.stop()
        return True
    else:
        # No path found
        stats.stop()
        return False

# ===============================
# RESET
# ===============================
def reset_grid(grid, start, end):
    for row in grid:
        for node in row:
            if node not in (start, end):
                node.color = WHITE
            node.parent = None

# ===============================
# MAIN (Updated)
# ===============================
def main():
    grid = create_grid()
    start = grid[2][2]
    end = grid[15][15]
    start.color = GREEN
    end.color = BLUE

    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Limit frame rate to reduce blinking
        clock.tick(60)
        
        button_rects = draw(grid, WIN, stats)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                if button_rects["BFS"].collidepoint(pos):
                    reset_grid(grid, start, end)
                    if not bfs(grid, start, end, WIN):
                        show_no_path_message()
                    draw(grid, WIN, stats)

                elif button_rects["DFS"].collidepoint(pos):
                    reset_grid(grid, start, end)
                    if not dfs(grid, start, end, WIN):
                        show_no_path_message()
                    draw(grid, WIN, stats)

                elif button_rects["UCS"].collidepoint(pos):
                 reset_grid(grid, start, end)
                 result = ucs(grid, start, end, WIN)
                 if not result:
                  show_no_path_message()
                 draw(grid, WIN, stats)  # Force redraw after algorithm
                elif button_rects["DLS"].collidepoint(pos):
                    reset_grid(grid, start, end)
                    if not dls(grid, start, end, 10, WIN):
                        show_no_path_message()
                    draw(grid, WIN, stats)

                elif button_rects["IDDFS"].collidepoint(pos):
                    reset_grid(grid, start, end)
                    if not iddfs(grid, start, end, WIN):
                        # Message already shown in iddfs
                        pass
                    draw(grid, WIN, stats)

                elif button_rects["BIDIRECTIONAL"].collidepoint(pos):
                    reset_grid(grid, start, end)
                    if not bidirectional(grid, start, end, WIN):
                        show_no_path_message()
                    draw(grid, WIN, stats)

                elif button_rects["RESET"].collidepoint(pos):
                    reset_grid(grid, start, end)
                    stats.reset()
                    draw(grid, WIN, stats)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()