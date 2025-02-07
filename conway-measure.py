import numpy as np
import time
import matplotlib.pyplot as plt
import conway  # Assuming the original Conway's Game of Life script is named conway.py
import cProfile
from line_profiler import profile


def measure_execution_time(grid_sizes, iterations=10):
    """
    Measure execution time for different grid sizes

    Args:
    grid_sizes (list): List of grid sizes to test
    iterations (int): Number of iterations to run for each grid size

    Returns:
    tuple: Lists of grid sizes and corresponding execution times
    """
    execution_times = []

    for N in grid_sizes:
        # Create random grid
        grid = conway.randomGrid(N)

        # Measure execution time
        start_time = time.time()
        for _ in range(iterations):
            # Copy of the update function from conway.py, but without matplotlib
            newGrid = grid.copy()
            for i in range(N):
                for j in range(N):
                    total = int(
                        (
                                grid[i, (j - 1) % N]
                                + grid[i, (j + 1) % N]
                                + grid[(i - 1) % N, j]
                                + grid[(i + 1) % N, j]
                                + grid[(i - 1) % N, (j - 1) % N]
                                + grid[(i - 1) % N, (j + 1) % N]
                                + grid[(i + 1) % N, (j - 1) % N]
                                + grid[(i + 1) % N, (j + 1) % N]
                        )
                        / 255
                    )
                    if grid[i, j] == conway.ON:
                        if (total < 2) or (total > 3):
                            newGrid[i, j] = conway.OFF
                    else:
                        if total == 3:
                            newGrid[i, j] = conway.ON

                    grid[:] = newGrid[:]

        end_time = time.time()
        execution_times.append((end_time - start_time) / iterations)

    return grid_sizes, execution_times

@profile
def measure_execution_time_optimized(grid_sizes, iterations=10):
    """
    Measure execution time for different grid sizes

    Args:
    grid_sizes (list): List of grid sizes to test
    iterations (int): Number of iterations to run for each grid size

    Returns:
    tuple: Lists of grid sizes and corresponding execution times
    """
    execution_times = []

    for N in grid_sizes:
        # Create random grid
        grid = conway.randomGrid(N)

        # Measure execution time
        start_time = time.time()
        for _ in range(iterations):
            # Vectorized neighbor sum using convolution (kernel for 8 neighbors)
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            neighbors = np.zeros((N, N), dtype=int)

            for offset in range(1, 9):  # Directions to account for neighbors (cardinal + diagonal)
                # Generate the shifted grids for each direction and sum them
                shifted_grid = np.roll(grid, offset, axis=0)  # Shifting along axis
                neighbors += shifted_grid

            # Update the grid based on the neighbor sum
            newGrid = grid.copy()
            total_neighbors = neighbors / 255  # Scale it to match the grid ON/OFF state

            # Apply Conway's Game of Life rules
            for i in range(N):
                for j in range(N):
                    total = total_neighbors[i, j]
                    if grid[i, j] == conway.ON:
                        if total < 2 or total > 3:
                            newGrid[i, j] = conway.OFF
                    else:
                        if total == 3:
                            newGrid[i, j] = conway.ON

            # Update grid in place after all iterations
            grid[:] = newGrid[:]

        end_time = time.time()
        execution_times.append((end_time - start_time) / iterations)

    return grid_sizes, execution_times


def plot_performance(grid_sizes, execution_times_original, execution_times_optimized):
    """
    Plot execution times against grid sizes for both original and optimized methods

    Args:
    grid_sizes (list): List of grid sizes tested
    execution_times_original (list): Execution times for the original method
    execution_times_optimized (list): Execution times for the optimized method
    """
    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, execution_times_original, marker='o', label='Original Method', color='blue')
    plt.plot(grid_sizes, execution_times_optimized, marker='x', label='Optimized Method', color='red')
    plt.title("Execution Time vs Grid Size")
    plt.xlabel("Grid Size (N)")
    plt.ylabel("Average Execution Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    sizes = [10, 25, 50, 75, 100, 150, 200, 250, 300]

    # Measure execution time for original method
    grid_sizes_original, times_original = measure_execution_time(sizes)

    # Measure execution time for optimized method
    grid_sizes_optimized, times_optimized = measure_execution_time_optimized(sizes)

    # Plot performance comparison
    plot_performance(grid_sizes_original, times_original, times_optimized)

    # Print results for reference
    print("Original Method:")
    for size, time in zip(grid_sizes_original, times_original):
        print(f"Grid Size: {size}x{size}, Average Execution Time: {time:.4f} seconds")

    print("\nOptimized Method:")
    for size, time in zip(grid_sizes_optimized, times_optimized):
        print(f"Grid Size: {size}x{size}, Average Execution Time: {time:.4f} seconds")