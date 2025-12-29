import cv2
import numpy as np
import heapq
import os
import math

# --- Helper Functions ---

def find_color_center(image, lower_bound, upper_bound):
    """Finds the centroid of a specified color in an image."""
    mask = cv2.inRange(image, lower_bound, upper_bound)
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    # Calculate centroid coordinates
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cy, cx)  # Return as (row, col) for numpy array indexing

def heuristic(a, b):
    """Calculates the Euclidean distance heuristic for A*."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def reconstruct_path(came_from, current):
    """Reconstructs the path from the came_from map."""
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path

def a_star_search(maze_grid, cost_map, start, end):
    """
    Performs A* search to find the lowest-cost path from start to end.
    The cost function is designed to prefer paths away from walls.
    """
    rows, cols = maze_grid.shape
    
    # Priority queue for nodes to visit: (f_score, node)
    open_set = [(0, start)]
    # Set to keep track of visited nodes
    closed_set = set()
    
    # Dictionary to reconstruct the path
    came_from = {}
    
    # g_score: cost from start to the current node
    g_score = np.full((rows, cols), float('inf'), dtype=np.float32)
    g_score[start] = 0
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        if current == end:
            return reconstruct_path(came_from, current)

        (r, c) = current
        # Explore 8-directional neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (r + dr, c + dc)
            nr, nc = neighbor
            
            # Check if neighbor is within bounds and is a valid path
            if not (0 <= nr < rows and 0 <= nc < cols and maze_grid[nr, nc] != 0):
                continue
            
            # Calculate the cost to move to the neighbor
            step_dist = math.sqrt(dr**2 + dc**2)
            # Cost function: base distance + a heavy penalty for being near a wall
            # The cost_map value is higher near walls, so we scale it to make it significant.
            move_cost = (1 + cost_map[neighbor] * 50) * step_dist
            
            tentative_g_score = g_score[r, c] + move_cost

            if tentative_g_score < g_score[nr, nc]:
                came_from[neighbor] = current
                g_score[nr, nc] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score, neighbor))
                    
    return None  # No path found

# --- Main Solver Function ---

def solve_maze(image_path, output_path):
    """
    Solves a maze from an image file, drawing the solution path.
    The path is calculated to stay in the middle of the corridors.
    """
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Identify start (red) and end (green) points
    # Color bounds in BGR format
    red_lower = np.array([0, 0, 150])
    red_upper = np.array([100, 100, 255])
    green_lower = np.array([0, 150, 0])
    green_upper = np.array([100, 255, 100])

    start_node = find_color_center(image, red_lower, red_upper)
    end_node = find_color_center(image, green_lower, green_upper)

    if start_node is None or end_node is None:
        print("Error: Could not find start (red) or end (green) point.")
        return
    print(f"Start point found at (row, col): {start_node}")
    print(f"End point found at (row, col): {end_node}")

    # 3. Preprocess the image for pathfinding
    # Create a copy to avoid modifying the original before drawing the path
    clean_image = image.copy()
    
    # Erase the start/end markers and any nearby artifacts by drawing white circles
    # This ensures a clear path for the algorithm
    cv2.circle(clean_image, (start_node[1], start_node[0]), 30, (255, 255, 255), -1)
    cv2.circle(clean_image, (end_node[1], end_node[0]), 30, (255, 255, 255), -1)

    # Convert to grayscale and create a binary grid (0=wall, 255=path)
    gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    _, maze_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 4. Create a cost map to guide the path to the center
    # Distance transform: for each path pixel, find its distance to the nearest wall
    dist_transform = cv2.distanceTransform(maze_binary, cv2.DIST_L2, 5)
    
    # Normalize the distance transform to a [0, 1] range
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # Create cost map: cost is high near walls (low distance) and low in the center (high distance)
    # Squaring the inverted distance heavily penalizes proximity to walls.
    cost_map = (1.0 - dist_transform)**2
    
    # The grid for A* where 1 is path and 0 is wall
    maze_grid = (maze_binary > 0).astype(np.uint8)

    # 5. Find the path using A* search
    print("Searching for a path...")
    path = a_star_search(maze_grid, cost_map, start_node, end_node)

    # 6. Draw the solution on the original image
    if path:
        print(f"Path found with {len(path)} points.")
        # Convert path from (row, col) to (x, y) for drawing
        path_points = np.array([[c, r] for r, c in path], dtype=np.int32)
        
        # Draw a thick, blue, anti-aliased line (BGR for blue is 255, 0, 0)
        cv2.polylines(image, [path_points], isClosed=False, color=(255, 0, 0), thickness=5, lineType=cv2.LINE_AA)
    else:
        print("No path could be found.")

    # 7. Save the final image
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, image)
    print(f"Solution image saved to {output_path}")

# --- Execution Block ---
if __name__ == '__main__':
    # Define the input and output file paths as specified
    input_image_path = 'images/maze_inverted.png'
    output_image_path = 'images/maze_solution.png'
    
    # Run the maze solver
    solve_maze(input_image_path, output_image_path)