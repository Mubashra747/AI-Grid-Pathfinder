"""
============================================================
  GOOD PERFORMANCE TIME APP — Pathfinder Visualizer
  Uninformed Search Algorithms on a 20x20 Grid
  
  Algorithms: BFS, DFS, UCS, DLS, IDDFS, Bidirectional
  Fixes applied:
    - DFS: mark visited on push
    - IDDFS: reset parents BEFORE each depth iteration
    - All algorithms: event loop added to prevent freezing
    - Fonts cached at module level (not recreated each frame)
    - Magic numbers replaced with named constants
    - Global stats removed; passed explicitly as parameter
    - is_obstacle() helper removed; access attribute directly
    - Neighbour pre-computation added for performance
============================================================
"""

import pygame
import sys
import heapq
from collections import deque
from typing import Optional

# ──────────────────────────────────────────────
# INITIALISE PYGAME
# ──────────────────────────────────────────────
pygame.init()

# ──────────────────────────────────────────────
# WINDOW / GRID CONSTANTS
# ──────────────────────────────────────────────
GRID_PIXEL_SIZE   = 600          # Width of the grid area in pixels
SIDE_PANEL_WIDTH  = 220          # Width of the right-side button panel
WINDOW_WIDTH      = GRID_PIXEL_SIZE + SIDE_PANEL_WIDTH
WINDOW_HEIGHT     = 600
NUM_ROWS          = 20           # Grid is NUM_ROWS × NUM_ROWS
CELL_PIXEL_SIZE   = GRID_PIXEL_SIZE // NUM_ROWS

# ──────────────────────────────────────────────
# ALGORITHM / ANIMATION CONSTANTS
# ──────────────────────────────────────────────
STEP_DELAY_MS        = 30    # Milliseconds between each algorithm step
PATH_DRAW_DELAY_MS   = 30    # Milliseconds between each path cell being coloured
PATH_PAUSE_MS        = 500   # Pause after full path is drawn
NO_PATH_PAUSE_MS     = 1500  # How long "NO PATH FOUND" message stays on screen
IDDFS_STEP_DELAY_MS  = 20    # Slightly faster delay for IDDFS inner loop
DLS_DEPTH_LIMIT      = 10    # Default depth limit for Depth-Limited Search
DIAGONAL_MOVE_COST   = 1.4   # Approximate √2, used in UCS for diagonal edges
CARDINAL_MOVE_COST   = 1.0   # Cost for up/down/left/right moves in UCS

# ──────────────────────────────────────────────
# COLOURS  (RGB)
# ──────────────────────────────────────────────
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0  )
GREEN   = (0,   200, 0  )    # Start node
BLUE    = (0,   100, 255)    # End node
RED     = (220, 60,  60 )    # Explored / visited node
YELLOW  = (255, 220, 0  )    # Frontier node (in queue/stack)
PURPLE  = (128, 0,   160)    # Final path
GRAY    = (200, 200, 200)    # UI buttons / panels
BROWN   = (139, 69,  19 )    # Obstacle / wall

# ──────────────────────────────────────────────
# MOVEMENT DIRECTIONS  (clockwise, assignment spec)
# Order: Up, Right, Down, Down-Right (diagonal), Left, Up-Left (diagonal)
# ──────────────────────────────────────────────
MOVEMENT_DIRECTIONS = [
    (-1,  0),   # 1. Up
    ( 0,  1),   # 2. Right
    ( 1,  0),   # 3. Down
    ( 1,  1),   # 4. Down-Right (diagonal)
    ( 0, -1),   # 5. Left
    (-1, -1),   # 6. Up-Left   (diagonal)
]

# ──────────────────────────────────────────────
# FONTS  (cached once — never inside draw loops)
# ──────────────────────────────────────────────
FONT_SMALL  = pygame.font.SysFont(None, 20)
FONT_MEDIUM = pygame.font.SysFont(None, 24)
FONT_LARGE  = pygame.font.SysFont(None, 48)

# ──────────────────────────────────────────────
# WINDOW
# ──────────────────────────────────────────────
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("GOOD PERFORMANCE TIME APP")


# ================================================================
# STATS TRACKER
# Tracks algorithm name, nodes explored, and elapsed time.
# Draws a small info box in the bottom-right of the side panel.
# ================================================================
class StatsTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.algorithm_name  = ""
        self.nodes_explored  = 0
        self.start_time_ms   = 0
        self.end_time_ms     = 0
        self.is_visible      = False

    def start_tracking(self, algorithm_name: str) -> None:
        """Call this just before running an algorithm."""
        self.reset()
        self.algorithm_name = algorithm_name
        self.start_time_ms  = pygame.time.get_ticks()
        self.is_visible     = True

    def stop_tracking(self) -> None:
        """Call this after the algorithm finishes."""
        self.end_time_ms = pygame.time.get_ticks()

    def record_node(self) -> None:
        """Increment the explored-node counter by one."""
        self.nodes_explored += 1

    def get_elapsed_ms(self) -> int:
        if self.end_time_ms > 0:
            return self.end_time_ms - self.start_time_ms
        return 0

    def draw(self, surface: pygame.Surface) -> None:
        if not self.is_visible or not self.algorithm_name:
            return

        box_rect = pygame.Rect(GRID_PIXEL_SIZE + 10, WINDOW_HEIGHT - 85, 200, 80)
        pygame.draw.rect(surface, GRAY,  box_rect)
        pygame.draw.rect(surface, BLACK, box_rect, 2)

        algo_label = FONT_MEDIUM.render(f"Algorithms: {self.algorithm_name}", True, BLACK)
        nodes_label = FONT_MEDIUM.render(f"Nodes: {self.nodes_explored}",  True, BLACK)
        time_label  = FONT_MEDIUM.render(f"Time:  {self.get_elapsed_ms()}ms", True, BLACK)

        surface.blit(algo_label,  (GRID_PIXEL_SIZE + 15, WINDOW_HEIGHT - 78))
        surface.blit(nodes_label, (GRID_PIXEL_SIZE + 15, WINDOW_HEIGHT - 53))
        surface.blit(time_label,  (GRID_PIXEL_SIZE + 15, WINDOW_HEIGHT - 28))


# ================================================================
# NODE CLASS
# Represents one cell in the grid.
# ================================================================
class Node:
    def __init__(self, row: int, col: int) -> None:
        self.row         = row
        self.col         = col
        self.color       = WHITE
        self.parent: Optional["Node"] = None
        self.is_obstacle = False
        self.neighbours: list["Node"] = []   # pre-computed; filled by build_neighbour_lists()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(
            surface,
            self.color,
            (self.col * CELL_PIXEL_SIZE, self.row * CELL_PIXEL_SIZE,
             CELL_PIXEL_SIZE, CELL_PIXEL_SIZE),
        )


# ================================================================
# GRID HELPERS
# ================================================================
def create_grid() -> list[list[Node]]:
    """Return a fresh NUM_ROWS × NUM_ROWS grid of white Nodes."""
    return [[Node(row, col) for col in range(NUM_ROWS)] for row in range(NUM_ROWS)]


def build_neighbour_lists(grid: list[list[Node]]) -> None:
    """
    Pre-compute the valid (in-bounds) neighbours for every node.
    Call once after grid creation (or after ROWS changes).
    This avoids repeating the bounds-check inside every algorithm step.
    """
    for grid_row in grid:
        for node in grid_row:
            node.neighbours = []
            for row_delta, col_delta in MOVEMENT_DIRECTIONS:
                neighbour_row = node.row + row_delta
                neighbour_col = node.col + col_delta
                if 0 <= neighbour_row < NUM_ROWS and 0 <= neighbour_col < NUM_ROWS:
                    node.neighbours.append(grid[neighbour_row][neighbour_col])


def reset_search_state(grid: list[list[Node]], start_node: Node, end_node: Node) -> None:
    """
    Clear visited/path colours and parent pointers from a previous run.
    Obstacles and the start/end colours are preserved.
    """
    for grid_row in grid:
        for node in grid_row:
            if node not in (start_node, end_node) and not node.is_obstacle:
                node.color = WHITE
            node.parent = None


def toggle_obstacle(
    grid: list[list[Node]],
    row: int,
    col: int,
    start_node: Node,
    end_node: Node,
) -> None:
    """Toggle a cell between obstacle and open. Protects start/end nodes."""
    cell = grid[row][col]
    if cell in (start_node, end_node):
        return
    cell.is_obstacle = not cell.is_obstacle
    cell.color = BROWN if cell.is_obstacle else WHITE


def clear_all_obstacles(grid: list[list[Node]], start_node: Node, end_node: Node) -> None:
    """Remove every obstacle from the grid."""
    for grid_row in grid:
        for node in grid_row:
            if node not in (start_node, end_node):
                node.is_obstacle = False
                node.color = WHITE


# ================================================================
# DRAWING
# ================================================================
def draw_grid_lines(surface: pygame.Surface) -> None:
    for index in range(NUM_ROWS + 1):
        pygame.draw.line(surface, BLACK, (0, index * CELL_PIXEL_SIZE), (GRID_PIXEL_SIZE, index * CELL_PIXEL_SIZE))
        pygame.draw.line(surface, BLACK, (index * CELL_PIXEL_SIZE, 0), (index * CELL_PIXEL_SIZE, GRID_PIXEL_SIZE))


def draw_side_panel(surface: pygame.Surface) -> dict[str, pygame.Rect]:
    """
    Render all buttons in the side panel.
    Returns a dict mapping button name → its pygame.Rect (for click detection).
    """
    button_width   = 180
    button_height  = 40
    button_margin  = 15
    panel_start_x  = GRID_PIXEL_SIZE + 20
    panel_start_y  = 20

    button_names = ["BFS", "DFS", "UCS", "DLS", "IDDFS", "BIDIRECTIONAL", "RESET", "CLEAR OBSTACLES"]
    button_rects: dict[str, pygame.Rect] = {}

    for index, name in enumerate(button_names):
        button_y    = panel_start_y + index * (button_height + button_margin)
        button_rect = pygame.Rect(panel_start_x, button_y, button_width, button_height)
        pygame.draw.rect(surface, GRAY,  button_rect)
        pygame.draw.rect(surface, BLACK, button_rect, 2)
        label = FONT_MEDIUM.render(name, True, BLACK)
        surface.blit(label, (panel_start_x + 28, button_y + 11))
        button_rects[name] = button_rect

    # Usage instructions below the buttons
    hint_y = panel_start_y + len(button_names) * (button_height + button_margin) + 10
    for line in ["Click grid to place /", "remove obstacles.", "(Brown = obstacle)"]:
        hint_surface = FONT_SMALL.render(line, True, BLACK)
        surface.blit(hint_surface, (GRID_PIXEL_SIZE + 25, hint_y))
        hint_y += 20

    return button_rects


def draw_everything(
    grid: list[list[Node]],
    surface: pygame.Surface,
    stats: StatsTracker,
) -> dict[str, pygame.Rect]:
    """Full redraw of the window. Returns the button rects for click detection."""
    surface.fill(WHITE)
    for grid_row in grid:
        for node in grid_row:
            node.draw(surface)
    draw_grid_lines(surface)
    button_rects = draw_side_panel(surface)
    stats.draw(surface)
    pygame.display.update()
    return button_rects


def show_no_path_message() -> None:
    """Overlay a semi-transparent 'NO PATH FOUND' message over the grid."""
    overlay = pygame.Surface((GRID_PIXEL_SIZE, WINDOW_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(WHITE)
    WINDOW.blit(overlay, (0, 0))

    message = FONT_LARGE.render("NO PATH FOUND!", True, RED)
    message_rect = message.get_rect(center=(GRID_PIXEL_SIZE // 2, WINDOW_HEIGHT // 2 - 40))
    WINDOW.blit(message, message_rect)

    pygame.display.update()
    pygame.time.delay(NO_PATH_PAUSE_MS)


# ================================================================
# PATH RECONSTRUCTION HELPERS
# ================================================================
def reconstruct_and_draw_path(
    start_node: Node,
    end_node: Node,
    grid: list[list[Node]],
    surface: pygame.Surface,
    stats: StatsTracker,
) -> None:
    """
    Walk the parent chain from end → start, collect intermediate nodes,
    then colour them purple one by one (start and end keep their colours).
    """
    path_nodes: list[Node] = []
    current = end_node

    while current != start_node:
        current = current.parent
        if current is None:
            return          # Safety: broken parent chain
        if current != start_node:
            path_nodes.append(current)

    for node in reversed(path_nodes):
        node.color = PURPLE
        draw_everything(grid, surface, stats)
        pygame.time.delay(PATH_DRAW_DELAY_MS)

    pygame.time.delay(PATH_PAUSE_MS)


def reconstruct_and_draw_bidirectional_path(
    start_node: Node,
    meeting_node: Node,
    end_node: Node,
    parent_from_start: dict[Node, Optional[Node]],
    parent_from_end: dict[Node, Optional[Node]],
    grid: list[list[Node]],
    surface: pygame.Surface,
    stats: StatsTracker,
) -> None:
    """
    Reconstruct path for Bidirectional Search.
    Walks from meeting node → start (using forward parents)
    and from meeting node → end (using backward parents).
    """
    # Segment 1: meeting → start
    segment_to_start: list[Node] = []
    current = meeting_node
    while current != start_node:
        current = parent_from_start.get(current)
        if current is None:
            break
        segment_to_start.append(current)

    # Segment 2: meeting → end
    segment_to_end: list[Node] = []
    current = meeting_node
    while current != end_node:
        current = parent_from_end.get(current)
        if current is None:
            break
        segment_to_end.append(current)

    for node in reversed(segment_to_start):
        if node not in (start_node, meeting_node):
            node.color = PURPLE
        draw_everything(grid, surface, stats)
        pygame.time.delay(PATH_DRAW_DELAY_MS)

    for node in segment_to_end:
        if node not in (end_node, meeting_node):
            node.color = PURPLE
        draw_everything(grid, surface, stats)
        pygame.time.delay(PATH_DRAW_DELAY_MS)

    pygame.time.delay(PATH_PAUSE_MS)


# ================================================================
# ALGORITHM WRAPPER
# Handles: grid reset, stats start/stop, no-path message.
# ================================================================
def run_algorithm(
    algorithm_function,
    grid: list[list[Node]],
    start_node: Node,
    end_node: Node,
    surface: pygame.Surface,
    stats: StatsTracker,
    display_name: str,
    *extra_args,
) -> bool:
    """
    Unified wrapper for all search algorithms.
    - Resets the grid before running.
    - Starts and stops the stats tracker.
    - Shows 'NO PATH FOUND' if the algorithm returns False.
    """
    reset_search_state(grid, start_node, end_node)
    stats.start_tracking(display_name)

    if extra_args:
        path_found = algorithm_function(grid, start_node, end_node, surface, stats, *extra_args)
    else:
        path_found = algorithm_function(grid, start_node, end_node, surface, stats)

    stats.stop_tracking()

    if not path_found:
        reset_search_state(grid, start_node, end_node)
        draw_everything(grid, surface, stats)
        show_no_path_message()

    draw_everything(grid, surface, stats)
    return path_found


# ──────────────────────────────────────────────
# SHARED HELPER: poll events inside algorithm loops
# ──────────────────────────────────────────────
def handle_quit_events() -> None:
    """Check for QUIT events so the window stays responsive during search."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


# ================================================================
# ALGORITHM 1 — BREADTH-FIRST SEARCH (BFS)
# Explores layer by layer. Guarantees the shortest path (fewest hops).
# ================================================================
def breadth_first_search(
    grid: list[list[Node]],
    start_node: Node,
    end_node: Node,
    surface: pygame.Surface,
    stats: StatsTracker,
) -> bool:
    # Queue holds nodes to explore. We mark visited ON ENQUEUE to avoid duplicates.
    frontier_queue: deque[Node] = deque([start_node])
    visited_positions: set[tuple[int, int]] = {(start_node.row, start_node.col)}

    while frontier_queue:
        handle_quit_events()
        pygame.time.delay(STEP_DELAY_MS)

        current_node = frontier_queue.popleft()
        stats.record_node()

        if current_node == end_node:
            reconstruct_and_draw_path(start_node, end_node, grid, surface, stats)
            return True

        for neighbour in current_node.neighbours:
            neighbour_position = (neighbour.row, neighbour.col)

            if neighbour_position not in visited_positions and not neighbour.is_obstacle:
                visited_positions.add(neighbour_position)
                neighbour.parent = current_node

                if neighbour != end_node:
                    neighbour.color = YELLOW   # frontier colour

                frontier_queue.append(neighbour)

        # Mark current node as fully explored
        if current_node not in (start_node, end_node):
            current_node.color = RED

        draw_everything(grid, surface, stats)

    return False   # No path found


# ================================================================
# ALGORITHM 2 — DEPTH-FIRST SEARCH (DFS)
# Explores as deep as possible before backtracking.
# Does NOT guarantee shortest path.
# FIX: visited set is updated on PUSH (not pop) to prevent duplicates.
# ================================================================
def depth_first_search(
    grid: list[list[Node]],
    start_node: Node,
    end_node: Node,
    surface: pygame.Surface,
    stats: StatsTracker,
) -> bool:
    # Mark start as visited immediately so it's never re-added
    frontier_stack: list[Node] = [start_node]
    visited_positions: set[tuple[int, int]] = {(start_node.row, start_node.col)}

    while frontier_stack:
        handle_quit_events()
        pygame.time.delay(STEP_DELAY_MS)

        current_node = frontier_stack.pop()
        stats.record_node()

        if current_node == end_node:
            reconstruct_and_draw_path(start_node, end_node, grid, surface, stats)
            return True

        for neighbour in current_node.neighbours:
            neighbour_position = (neighbour.row, neighbour.col)

            # ✅ FIX: mark visited HERE (on push) — not after popping
            if neighbour_position not in visited_positions and not neighbour.is_obstacle:
                visited_positions.add(neighbour_position)   # prevent future re-pushes
                neighbour.parent = current_node

                if neighbour != end_node:
                    neighbour.color = YELLOW

                frontier_stack.append(neighbour)

        if current_node not in (start_node, end_node):
            current_node.color = RED

        draw_everything(grid, surface, stats)

    return False


# ================================================================
# ALGORITHM 3 — UNIFORM-COST SEARCH (UCS)
# Expands the lowest-cost node first.
# Uses cost 1.0 for cardinal moves and 1.4 (≈√2) for diagonals.
# ================================================================
def uniform_cost_search(
    grid: list[list[Node]],
    start_node: Node,
    end_node: Node,
    surface: pygame.Surface,
    stats: StatsTracker,
) -> bool:
    # tie_breaker ensures stable heap ordering when costs are equal
    tie_breaker = 0
    priority_heap: list[tuple[float, int, Node]] = []
    heapq.heappush(priority_heap, (0.0, tie_breaker, start_node))
    tie_breaker += 1

    # best known cost to reach each position
    best_cost_to: dict[tuple[int, int], float] = {(start_node.row, start_node.col): 0.0}
    # parent map for path reconstruction
    parent_map: dict[Node, Optional[Node]] = {start_node: None}
    # fully settled nodes
    settled_positions: set[tuple[int, int]] = set()

    while priority_heap:
        handle_quit_events()
        pygame.time.delay(STEP_DELAY_MS)

        current_cost, _, current_node = heapq.heappop(priority_heap)
        current_position = (current_node.row, current_node.col)

        if current_position in settled_positions:
            continue    # already processed via a cheaper path

        settled_positions.add(current_position)
        stats.record_node()

        if current_node not in (start_node, end_node):
            current_node.color = RED

        if current_node == end_node:
            # Reconstruct path directly from parent_map
            path_nodes: list[Node] = []
            node_cursor = end_node
            while node_cursor is not None:
                path_nodes.append(node_cursor)
                node_cursor = parent_map.get(node_cursor)

            for node in reversed(path_nodes):
                if node not in (start_node, end_node):
                    node.color = PURPLE
                draw_everything(grid, surface, stats)
                pygame.time.delay(PATH_DRAW_DELAY_MS)

            pygame.time.delay(PATH_PAUSE_MS)
            return True

        for neighbour in current_node.neighbours:
            if neighbour.is_obstacle:
                continue

            neighbour_position = (neighbour.row, neighbour.col)

            # Determine move cost based on whether it's a diagonal
            row_delta = abs(current_node.row - neighbour.row)
            col_delta = abs(current_node.col - neighbour.col)
            is_diagonal = (row_delta == 1 and col_delta == 1)
            move_cost = DIAGONAL_MOVE_COST if is_diagonal else CARDINAL_MOVE_COST

            new_total_cost = current_cost + move_cost

            if neighbour_position not in best_cost_to or new_total_cost < best_cost_to[neighbour_position]:
                best_cost_to[neighbour_position] = new_total_cost
                parent_map[neighbour] = current_node
                heapq.heappush(priority_heap, (new_total_cost, tie_breaker, neighbour))
                tie_breaker += 1

                if neighbour not in (start_node, end_node):
                    neighbour.color = YELLOW

        draw_everything(grid, surface, stats)

    return False


# ================================================================
# ALGORITHM 4 — DEPTH-LIMITED SEARCH (DLS)
# DFS that stops exploring beyond a fixed depth limit.
# ================================================================
def depth_limited_search(
    grid: list[list[Node]],
    start_node: Node,
    end_node: Node,
    surface: pygame.Surface,
    stats: StatsTracker,
    depth_limit: int = DLS_DEPTH_LIMIT,
) -> bool:
    # Stack stores (node, current_depth_level)
    frontier_stack: list[tuple[Node, int]] = [(start_node, 0)]
    visited_positions: set[tuple[int, int]] = {(start_node.row, start_node.col)}

    while frontier_stack:
        handle_quit_events()
        pygame.time.delay(STEP_DELAY_MS)

        current_node, current_depth = frontier_stack.pop()
        stats.record_node()

        if current_node == end_node:
            reconstruct_and_draw_path(start_node, end_node, grid, surface, stats)
            return True

        # Only expand this node if we haven't hit the depth limit
        if current_depth < depth_limit:
            for neighbour in current_node.neighbours:
                neighbour_position = (neighbour.row, neighbour.col)

                if neighbour_position not in visited_positions and not neighbour.is_obstacle:
                    visited_positions.add(neighbour_position)
                    neighbour.parent = current_node

                    if neighbour != end_node:
                        neighbour.color = YELLOW

                    frontier_stack.append((neighbour, current_depth + 1))

        if current_node not in (start_node, end_node):
            current_node.color = RED

        draw_everything(grid, surface, stats)

    return False


# ================================================================
# ALGORITHM 5 — ITERATIVE DEEPENING DFS (IDDFS)
# Runs DLS repeatedly with increasing depth limits (0, 1, 2, …)
# until the goal is found. Combines BFS's optimality with DFS's memory use.
#
# FIX: Parents are reset BEFORE each depth iteration (not after),
#      so parent pointers from earlier passes never contaminate the current one.
# ================================================================
def iterative_deepening_dfs(
    grid: list[list[Node]],
    start_node: Node,
    end_node: Node,
    surface: pygame.Surface,
    stats: StatsTracker,
) -> bool:
    max_possible_depth = NUM_ROWS * NUM_ROWS   # worst-case upper bound

    for current_depth_limit in range(max_possible_depth):

        # ✅ FIX: Reset visual state and parent pointers BEFORE this depth pass
        for grid_row in grid:
            for node in grid_row:
                if node not in (start_node, end_node) and not node.is_obstacle:
                    node.color = WHITE
                node.parent = None

        draw_everything(grid, surface, stats)

        # Run one depth-limited pass
        frontier_stack: list[tuple[Node, int]] = [(start_node, 0)]
        visited_positions: set[tuple[int, int]] = {(start_node.row, start_node.col)}
        goal_found = False

        while frontier_stack:
            handle_quit_events()
            pygame.time.delay(IDDFS_STEP_DELAY_MS)

            current_node, current_depth = frontier_stack.pop()
            stats.record_node()

            if current_node == end_node:
                goal_found = True
                break

            if current_depth < current_depth_limit:
                for neighbour in current_node.neighbours:
                    neighbour_position = (neighbour.row, neighbour.col)

                    if neighbour_position not in visited_positions and not neighbour.is_obstacle:
                        visited_positions.add(neighbour_position)
                        neighbour.parent = current_node

                        if neighbour != end_node:
                            neighbour.color = YELLOW

                        frontier_stack.append((neighbour, current_depth + 1))

            if current_node not in (start_node, end_node):
                current_node.color = RED

            draw_everything(grid, surface, stats)

        if goal_found:
            reconstruct_and_draw_path(start_node, end_node, grid, surface, stats)
            return True

    return False


# ================================================================
# ALGORITHM 6 — BIDIRECTIONAL SEARCH
# Runs BFS simultaneously from start AND from end.
# Stops when the two frontiers meet. Often explores far fewer nodes.
# ================================================================
def bidirectional_search(
    grid: list[list[Node]],
    start_node: Node,
    end_node: Node,
    surface: pygame.Surface,
    stats: StatsTracker,
) -> bool:
    if start_node == end_node:
        return True

    # Forward frontier (from start)
    forward_queue: deque[Node] = deque([start_node])
    forward_visited: dict[tuple[int, int], Node] = {(start_node.row, start_node.col): start_node}
    forward_parent: dict[Node, Optional[Node]] = {start_node: None}

    # Backward frontier (from end)
    backward_queue: deque[Node] = deque([end_node])
    backward_visited: dict[tuple[int, int], Node] = {(end_node.row, end_node.col): end_node}
    backward_parent: dict[Node, Optional[Node]] = {end_node: None}

    meeting_node: Optional[Node] = None

    while forward_queue and backward_queue and meeting_node is None:
        handle_quit_events()
        pygame.time.delay(STEP_DELAY_MS)

        # ── Expand one node from the forward frontier ──
        current_forward = forward_queue.popleft()
        stats.record_node()

        for neighbour in current_forward.neighbours:
            if neighbour.is_obstacle:
                continue

            neighbour_position = (neighbour.row, neighbour.col)

            # Check if this neighbour has already been reached from the other side
            if neighbour_position in backward_visited:
                meeting_node = neighbour
                forward_parent[neighbour] = current_forward
                break

            if neighbour_position not in forward_visited:
                forward_visited[neighbour_position] = neighbour
                forward_parent[neighbour] = current_forward

                if neighbour != end_node:
                    neighbour.color = YELLOW

                forward_queue.append(neighbour)

        if current_forward not in (start_node, end_node):
            current_forward.color = RED

        draw_everything(grid, surface, stats)

        if meeting_node:
            break

        # ── Expand one node from the backward frontier ──
        current_backward = backward_queue.popleft()
        stats.record_node()

        for neighbour in current_backward.neighbours:
            if neighbour.is_obstacle:
                continue

            neighbour_position = (neighbour.row, neighbour.col)

            if neighbour_position in forward_visited:
                meeting_node = neighbour
                backward_parent[neighbour] = current_backward
                break

            if neighbour_position not in backward_visited:
                backward_visited[neighbour_position] = neighbour
                backward_parent[neighbour] = current_backward

                if neighbour != start_node:
                    neighbour.color = YELLOW

                backward_queue.append(neighbour)

        if current_backward not in (end_node, start_node):
            current_backward.color = RED

        draw_everything(grid, surface, stats)

    # ── Path found: reconstruct from meeting point ──
    if meeting_node:
        # Briefly highlight the meeting node
        meeting_node.color = YELLOW
        draw_everything(grid, surface, stats)
        pygame.time.delay(PATH_PAUSE_MS)

        reconstruct_and_draw_bidirectional_path(
            start_node, meeting_node, end_node,
            forward_parent, backward_parent,
            grid, surface, stats,
        )
        return True

    return False


# ================================================================
# MAIN LOOP
# ================================================================
def main() -> None:
    grid = create_grid()
    build_neighbour_lists(grid)

    # Default start and end positions
    start_node = grid[2][2]
    end_node   = grid[15][15]
    start_node.color = GREEN
    end_node.color   = BLUE

    stats = StatsTracker()
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)
        button_rects = draw_everything(grid, WINDOW, stats)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # ── Grid area: toggle obstacles ──
                if mouse_x < GRID_PIXEL_SIZE and mouse_y < GRID_PIXEL_SIZE:
                    clicked_row = mouse_y // CELL_PIXEL_SIZE
                    clicked_col = mouse_x // CELL_PIXEL_SIZE
                    toggle_obstacle(grid, clicked_row, clicked_col, start_node, end_node)
                    draw_everything(grid, WINDOW, stats)

                # ── Side panel buttons ──
                mouse_pos = (mouse_x, mouse_y)

                if button_rects["BFS"].collidepoint(mouse_pos):
                    run_algorithm(breadth_first_search, grid, start_node, end_node, WINDOW, stats, "BFS")

                elif button_rects["DFS"].collidepoint(mouse_pos):
                    run_algorithm(depth_first_search, grid, start_node, end_node, WINDOW, stats, "DFS")

                elif button_rects["UCS"].collidepoint(mouse_pos):
                    run_algorithm(uniform_cost_search, grid, start_node, end_node, WINDOW, stats, "UCS")

                elif button_rects["DLS"].collidepoint(mouse_pos):
                    run_algorithm(depth_limited_search, grid, start_node, end_node, WINDOW, stats, f"DLS (limit={DLS_DEPTH_LIMIT})", DLS_DEPTH_LIMIT)

                elif button_rects["IDDFS"].collidepoint(mouse_pos):
                    run_algorithm(iterative_deepening_dfs, grid, start_node, end_node, WINDOW, stats, "IDDFS")

                elif button_rects["BIDIRECTIONAL"].collidepoint(mouse_pos):
                    run_algorithm(bidirectional_search, grid, start_node, end_node, WINDOW, stats, "BIDIRECTIONAL")

                elif button_rects["RESET"].collidepoint(mouse_pos):
                    reset_search_state(grid, start_node, end_node)
                    stats.reset()
                    draw_everything(grid, WINDOW, stats)

                elif button_rects["CLEAR OBSTACLES"].collidepoint(mouse_pos):
                    clear_all_obstacles(grid, start_node, end_node)
                    draw_everything(grid, WINDOW, stats)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
