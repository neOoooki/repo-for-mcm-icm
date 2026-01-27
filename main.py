import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d

class StairWearMVP:
    def __init__(self, length=1000, width=400, resolution=10,
                 walker_weight_kg=80, sliding_distance_mm=5,
                 hardness_mpa=200, wear_coefficient=1e-4):
        """
        Initialize the Stair Wear Model MVP.
        
        Args:
            length (float): Length of the stair in mm (X-axis, width of the staircase).
            width (float): Width/Depth of the stair in mm (Y-axis, tread depth).
            resolution (float): Grid resolution in mm.
            walker_weight_kg (float): Average weight of a walker in kg.
            sliding_distance_mm (float): Average sliding distance per step in mm.
            hardness_mpa (float): Hardness of the stair material in MPa (N/mm^2).
            wear_coefficient (float): Archard wear coefficient (dimensionless).
        """
        self.length = length
        self.width = width
        self.resolution = resolution
        
        # Physics parameters
        self.walker_weight_n = walker_weight_kg * 9.8  # Convert kg to N
        self.sliding_distance_mm = sliding_distance_mm
        self.hardness_mpa = hardness_mpa
        self.wear_coefficient = wear_coefficient
        
        # Grid dimensions (number of cells)
        # X corresponds to length (1000mm)
        # Y corresponds to width (400mm)
        self.nx = int(length / resolution)
        self.ny = int(width / resolution)
        
        # Initialize stair grid (wear depth), initially 0
        self.stair_grid = np.zeros((self.ny, self.nx))
        
        # Footprint params
        self.foot_length = 400
        self.foot_width = 100
        self.kernel = None
        self.single_step_wear_kernel = None # Actual wear depth map for one step

    def create_footprint_kernel(self):
        """
        Creates a footprint pressure kernel.
        Dimensions: 400mm (Y) x 100mm (X).
        """
        k_ny = int(self.foot_length / self.resolution) # 40
        k_nx = int(self.foot_width / self.resolution)  # 10
        
        # Create a coordinate grid for the kernel
        # Normalized coordinates -1 to 1
        y = np.linspace(-1, 1, k_ny)
        x = np.linspace(-1, 1, k_nx)
        X, Y = np.meshgrid(x, y)
        
        # Design a pressure distribution
        # Let's simulate a heel and a ball of the foot using Gaussians
        # Heel at bottom (negative Y), Ball at top (positive Y)
        
        # Heel
        Z_heel = np.exp(-((X)**2 + (Y + 0.5)**2) / 0.1)
        
        # Ball of foot
        Z_ball = np.exp(-((X)**2 + (Y - 0.5)**2) / 0.15)
        
        raw_kernel = Z_heel + Z_ball
        
        # Normalize raw kernel so that the sum of forces equals walker weight
        # Force = Pressure * Area
        # Total Force = Sum(P_ij * CellArea) = Weight
        # P_ij = Scale * raw_kernel_ij
        # Scale * Sum(raw_kernel_ij) * (res * res) = Weight
        
        cell_area_mm2 = self.resolution ** 2
        total_raw_sum = np.sum(raw_kernel)
        
        if total_raw_sum > 0:
            # Scale factor to convert raw kernel values to Pressure (N/mm^2 or MPa)
            scale_factor = self.walker_weight_n / (total_raw_sum * cell_area_mm2)
            self.kernel = raw_kernel * scale_factor
        else:
            self.kernel = raw_kernel

        # Calculate single step wear kernel based on Archard equation
        # Depth = (k / H) * Pressure * SlidingDist
        # Pressure is in MPa (N/mm^2), Hardness in MPa, Dist in mm -> Depth in mm
        
        if self.hardness_mpa > 0:
            wear_factor = (self.wear_coefficient / self.hardness_mpa) * self.sliding_distance_mm
            self.single_step_wear_kernel = self.kernel * wear_factor
        else:
            self.single_step_wear_kernel = np.zeros_like(self.kernel)

    def get_default_traffic_distribution(self, x_offset=0, y_offset=0, x_spread=200, y_spread=100):
        """
        Generates a default Gaussian traffic distribution map (probability density).
        Returns a grid of size (ny, nx) representing the probability of a step centering at each cell.
        """
        x = np.linspace(0, self.length, self.nx)
        y = np.linspace(0, self.width, self.ny)
        X, Y = np.meshgrid(x, y)
        
        # Center coordinates
        cx = (self.length / 2) + x_offset
        cy = (self.width / 2) + y_offset
        
        # Gaussian distribution
        dist = np.exp(-(((X - cx)**2) / (2 * x_spread**2) + ((Y - cy)**2) / (2 * y_spread**2)))
        
        # Normalize so sum is 1 (Probability Mass Function)
        total = np.sum(dist)
        if total > 0:
            dist = dist / total
            
        return dist

    def simulate_wear(self, years, people_per_year, 
                      traffic_dist_func=None, 
                      correction_func=None):
        """
        Simulates wear over a period.
        
        Args:
            years (float): Number of years.
            people_per_year (float): Number of people (footsteps) per year.
            traffic_dist_func (callable): Function returning (ny, nx) array of step probabilities.
            correction_func (callable): Function returning correction coefficient (scalar or array).
        """
        if self.single_step_wear_kernel is None:
            self.create_footprint_kernel()
            
        # 1. Determine Total Traffic Volume (Total Steps)
        total_steps = years * people_per_year
        
        # 2. Get Traffic Distribution (Where steps happen)
        if traffic_dist_func is None:
            # Default: Centered with some spread
            traffic_map_prob = self.get_default_traffic_distribution()
        else:
            traffic_map_prob = traffic_dist_func()
            
        # Convert probability map to actual step count map
        traffic_map_counts = traffic_map_prob * total_steps
        
        # 3. Apply Correction Coefficient
        if correction_func is not None:
            correction = correction_func()
            traffic_map_counts = traffic_map_counts * correction
            
        # 4. Convolve Traffic Map with Single Step Wear Kernel
        # We use 'same' to keep the grid size consistent, assuming steps near edges are handled by padding 0
        # However, for steps, the kernel (foot) has physical size.
        # If the 'traffic map' represents the CENTER of the foot, convolution is the correct operation
        # to accumulate wear from the footprint kernel placed at each center.
        
        total_wear = convolve2d(traffic_map_counts, self.single_step_wear_kernel, mode='same', boundary='fill', fillvalue=0)
        
        # Accumulate wear
        self.stair_grid += total_wear

    def visualize(self, filename='mvp_visualization.png'):
        """
        Visualizes the stair wear simulation results, including a 3D surface plot.
        """
        if self.single_step_wear_kernel is None:
            self.create_footprint_kernel()

        fig = plt.figure(figsize=(15, 18))
        
        # Grid layout: 3 rows, 2 columns
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])
        
        # 1. Wear Depth Map (2D)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.stair_grid, cmap='inferno_r', origin='lower', 
                           extent=[0, self.length, 0, self.width], aspect='equal')
        ax1.set_title(f"Stair Wear Depth (Max: {np.max(self.stair_grid):.4f} mm)")
        ax1.set_xlabel("Stair Length (mm)")
        ax1.set_ylabel("Stair Tread Depth (mm)")
        plt.colorbar(im1, ax=ax1, label='Wear Depth (mm)')
        
        # 2. Single Step Pressure/Wear Kernel
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.single_step_wear_kernel, cmap='plasma', origin='lower',
                           extent=[0, self.foot_width, 0, self.foot_length], aspect='equal')
        ax2.set_title(f"Single Step Wear Kernel (Max: {np.max(self.single_step_wear_kernel):.2e} mm)")
        ax2.set_xlabel("Foot Width (mm)")
        ax2.set_ylabel("Foot Length (mm)")
        plt.colorbar(im2, ax=ax2, label='Wear Depth per Step (mm)')
        
        # 3. Cross Section along Length (at mid-width)
        ax3 = fig.add_subplot(gs[1, 0])
        mid_y_idx = self.ny // 2
        ax3.plot(np.linspace(0, self.length, self.nx), self.stair_grid[mid_y_idx, :], label='Mid-Depth Profile')
        ax3.set_title("Cross Section: Wear along Length (Center)")
        ax3.set_xlabel("Stair Length (mm)")
        ax3.set_ylabel("Wear Depth (mm)")
        ax3.grid(True)
        ax3.invert_yaxis() # Depth goes down
        
        # 4. Cross Section along Width/Depth (at mid-length)
        ax4 = fig.add_subplot(gs[1, 1])
        mid_x_idx = self.nx // 2
        ax4.plot(np.linspace(0, self.width, self.ny), self.stair_grid[:, mid_x_idx], label='Mid-Length Profile', color='orange')
        ax4.set_title("Cross Section: Wear along Tread Depth (Center)")
        ax4.set_xlabel("Stair Tread Depth (mm)")
        ax4.set_ylabel("Wear Depth (mm)")
        ax4.grid(True)
        ax4.invert_yaxis()

        # 5. 3D Surface Plot
        ax5 = fig.add_subplot(gs[2, :], projection='3d')
        x = np.linspace(0, self.length, self.nx)
        y = np.linspace(0, self.width, self.ny)
        X, Y = np.meshgrid(x, y)
        Z = -self.stair_grid # 负值表示磨损深度
        
        surf = ax5.plot_surface(X, Y, Z, cmap='inferno_r', linewidth=0, antialiased=False)
        ax5.set_title("3D Stair Surface Model (Wear Exaggerated)")
        ax5.set_xlabel("Length (mm)")
        ax5.set_ylabel("Depth (mm)")
        ax5.set_zlabel("Height (mm)")
        
        # 设置3D图的长宽比与实际台阶比例一致
        # Z轴为了可视化效果进行夸张处理，设置为宽度的1/3
        ax5.set_box_aspect((self.length, self.width, self.width / 3))
        
        fig.colorbar(surf, ax=ax5, shrink=0.5, aspect=10, label='Surface Height (mm)')

        plt.tight_layout()
        plt.savefig(filename)
        print(f"Visualization saved to {os.path.abspath(filename)}")
        plt.close()

if __name__ == "__main__":
    print("Initializing Stair Wear MVP...")
    # Default: Stone (200 MPa), Rubber on Stone (k=1e-4), 80kg walker
    mvp = StairWearMVP(length=1000, width=400, resolution=10)
    
    print("Simulating Wear...")
    # Simulate 100 years, 500 people per day -> ~182,500 people/year
    # Total ~18 million steps
    years = 100
    people_per_day = 5000
    mvp.simulate_wear(years=years, people_per_year=people_per_day * 365)
    
    print(f"Max Wear Depth: {np.max(mvp.stair_grid):.4f} mm")
    
    print("Visualizing...")
    mvp.visualize()
    print("Done.")
