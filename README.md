# Pathfinder Visualizer — GOOD PERFORMANCE TIME APP

A step-by-step visual demonstration of six classic **uninformed search algorithms** navigating a 20×20 grid from a **Start** node to an **End** node, built with Python and Pygame.

---

## Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How to Use](#how-to-use)
- [Color Legend](#color-legend)
- [Movement Rules](#movement-rules)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)

---

## Overview

This project visualizes how different blind (uninformed) search algorithms explore a grid to find a path. Each algorithm animates step-by-step so you can watch exactly which nodes it checks, which it skips, and the final path it finds.

**Window title:** `GOOD PERFORMANCE TIME APP`  
**Grid size:** 20 × 20 cells  
**Start node:** top-left area (row 2, col 2) — shown in **green**  
**End node:** bottom-right area (row 15, col 15) — shown in **blue**

---

## Algorithms Implemented

| # | Algorithm | Shortest Path? | Notes |
|---|-----------|:--------------:|-------|
| 1 | **BFS** — Breadth-First Search | Yes (fewest hops) | Explores layer by layer |
| 2 | **DFS** — Depth-First Search | No | Explores as deep as possible first |
| 3 | **UCS** — Uniform-Cost Search |  Yes (lowest cost) | Uses cost 1.0 (cardinal) and 1.4 (diagonal) |
| 4 | **DLS** — Depth-Limited Search |  No | DFS capped at depth 10 |
| 5 | **IDDFS** — Iterative Deepening DFS |  Yes (fewest hops) | Runs DLS with increasing depth limits |
| 6 | **Bidirectional** |  Yes (fewest hops) | BFS from both start and end simultaneously |

---

## Requirements

- **Python** 3.10 or higher
- **pygame** 2.x

Check your Python version:

```bash
python --version
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/pathfinder-visualizer.git
cd pathfinder-visualizer
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install pygame
```

---

## How to Run

```bash
python pathfinder.py
```

The Pygame window will open immediately.

---

## How to Use

### Placing Obstacles
- **Left-click** any white cell on the grid to place a **brown obstacle** (wall).
- **Left-click** an existing obstacle to remove it.
- You cannot place obstacles on the Start (green) or End (blue) nodes.

### Running an Algorithm
Click any of the buttons in the right-side panel:

| Button | Action |
|--------|--------|
| `BFS` | Run Breadth-First Search |
| `DFS` | Run Depth-First Search |
| `UCS` | Run Uniform-Cost Search |
| `DLS` | Run Depth-Limited Search (limit = 10) |
| `IDDFS` | Run Iterative Deepening DFS |
| `BIDIRECTIONAL` | Run Bidirectional Search |
| `RESET` | Clear visited/path colours (keep obstacles) |
| `CLEAR OBSTACLES` | Remove all obstacles |

### Stats Panel
After each run a small stats box appears at the bottom-right showing:
- **Algorithm name**
- **Nodes explored**
- **Time elapsed (ms)**

---

## Color Legend

| Color | Meaning |
|-------|---------|
| **Green** | Start node |
| **Blue** | End node |
| **Yellow** | Frontier — node is in the queue/stack, waiting to be explored |
| **Red** | Explored — node has been fully visited |
| **Purple** | Final path from Start to End |
| **Brown** | Obstacle / wall |
| **White** | Unvisited open cell |

---

## Movement Rules

Each algorithm expands neighbours in this fixed **clockwise order**:

```
1. Up            (-1,  0)
2. Right         ( 0, +1)
3. Down          (+1,  0)
4. Down-Right    (+1, +1)  ← diagonal
5. Left          ( 0, -1)
6. Up-Left       (-1, -1)  ← diagonal
```

> **Note:** Top-Right and Bottom-Left diagonals are intentionally excluded per the assignment specification.

---

## Project Structure

```
pathfinder-visualizer/
│
├── pathfinder.py       # Main source file — all code lives here
└── README.md           # This file
```

### Key sections inside `pathfinder.py`

| Section | Description |
|---------|-------------|
| Constants | Window size, colours, delays, costs — all in one place at the top |
| `StatsTracker` | Tracks algorithm name, nodes explored, elapsed time |
| `Node` | Single grid cell with position, colour, parent pointer, neighbour list |
| Grid helpers | `create_grid`, `build_neighbour_lists`, `reset_search_state`, `toggle_obstacle` |
| Drawing | `draw_everything`, `draw_side_panel`, `show_no_path_message` |
| Path reconstruction | `reconstruct_and_draw_path`, `reconstruct_and_draw_bidirectional_path` |
| `run_algorithm` | Unified wrapper: resets grid, starts/stops stats, shows no-path message |
| Algorithm functions | `breadth_first_search`, `depth_first_search`, `uniform_cost_search`, `depth_limited_search`, `iterative_deepening_dfs`, `bidirectional_search` |
| `main` | Event loop, mouse handling, button routing |

---

## Known Limitations

- **DLS** may report "NO PATH FOUND" even when a path exists if the path length exceeds the depth limit (10 by default). Increase `DLS_DEPTH_LIMIT` in the constants section if needed.
- **IDDFS** re-explores nodes at every depth iteration, so it is visually slower than BFS even though it finds the same shortest path.
- The **Start** and **End** nodes are fixed at grid positions (2, 2) and (15, 15). They cannot be moved via the GUI in this version.
