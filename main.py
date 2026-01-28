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
        self.foot_length = 300
        self.foot_width = 100
        self.wear_kernel_up = None
        self.wear_kernel_down = None

    def init_kernels(self):
        """Initialize both up and down kernels."""
        _, self.wear_kernel_up = self.create_footprint_kernel(direction='up')
        _, self.wear_kernel_down = self.create_footprint_kernel(direction='down')

    def create_footprint_kernel(self, direction='up'):
        """
        Creates a footprint pressure kernel based on direction.
        Dimensions: 300mm (Y) x 100mm (X).
        
        Args:
            direction (str): 'up' or 'down'. 
                             'up': Higher pressure on ball (toes), impact factor 1.8.
                             'down': Higher pressure on heel, impact factor 3.0.
        Returns:
            tuple: (kernel, wear_kernel)
        """
        k_ny = int(self.foot_length / self.resolution) 
        k_nx = int(self.foot_width / self.resolution)
        
        # Create a coordinate grid for the kernel
        # Normalized coordinates -1 to 1
        y = np.linspace(-1, 1, k_ny)
        x = np.linspace(-1, 1, k_nx)
        X, Y = np.meshgrid(x, y)
        
        # Design a pressure distribution
        # Heel at bottom (negative Y), Ball at top (positive Y)
        # Base Gaussians
        Z_heel_base = np.exp(-((X)**2 + (Y + 0.5)**2) / 0.1)
        Z_ball_base = np.exp(-((X)**2 + (Y - 0.5)**2) / 0.15)
        
        impact_factor = 1.0
        
        if direction == 'up':
            # Up: Ball dominant (Front of foot)
            # Pressure larger at ball
            raw_kernel = 0.8 * Z_heel_base + 1.2 * Z_ball_base
            impact_factor = 1.8
            
        elif direction == 'down':
            # Down: Heel dominant
            # Pressure larger at heel
            raw_kernel = 1.5 * Z_heel_base + 0.8 * Z_ball_base
            impact_factor = 3.0
        else:
            raw_kernel = Z_heel_base + Z_ball_base
        
        # Normalize raw kernel so that the sum of forces equals walker weight
        cell_area_mm2 = self.resolution ** 2
        total_raw_sum = np.sum(raw_kernel)
        
        if total_raw_sum > 0:
            # Scale factor to convert raw kernel values to Pressure (N/mm^2 or MPa)
            scale_factor = (self.walker_weight_n * impact_factor) / (total_raw_sum * cell_area_mm2)
            kernel = raw_kernel * scale_factor
        else:
            kernel = raw_kernel

        # Calculate wear kernel
        if self.hardness_mpa > 0:
            wear_factor = (self.wear_coefficient / self.hardness_mpa) * self.sliding_distance_mm
            wear_kernel = kernel * wear_factor
        else:
            wear_kernel = np.zeros_like(kernel)
            
        return kernel, wear_kernel

    def get_single_traffic_distribution(self, x_offset=0, y_offset=0, x_spread=200, y_spread=100):
        """
        生成单人通行的足迹分布（单峰正态分布）。
        Generates single-file traffic distribution (Single Gaussian Peak).
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

    def get_parallel_traffic_distribution(self, separation=800, y_offset=0, x_spread=150, y_spread=100):
        """
        生成并行通行的足迹分布（双峰正态分布）。
        Generates parallel traffic distribution (Bi-modal Gaussian).
        
        Args:
            separation (float): Distance between two people in mm (along length/width depending on orientation).
                                Assuming separation is along the length (X-axis) for side-by-side walking.
            y_offset (float): Offset from center in Y direction.
            x_spread (float): Spread in X direction for each person.
            y_spread (float): Spread in Y direction.
        """
        x = np.linspace(0, self.length, self.nx)
        y = np.linspace(0, self.width, self.ny)
        X, Y = np.meshgrid(x, y)
        
        # Centers for two people walking side-by-side
        # Center of the stair
        cx_base = self.length / 2
        cy = (self.width / 2) + y_offset
        
        # Left and Right person positions
        cx_left = cx_base - (separation / 2)
        cx_right = cx_base + (separation / 2)
        
        # Gaussian distributions for both
        dist_left = np.exp(-(((X - cx_left)**2) / (2 * x_spread**2) + ((Y - cy)**2) / (2 * y_spread**2)))
        dist_right = np.exp(-(((X - cx_right)**2) / (2 * x_spread**2) + ((Y - cy)**2) / (2 * y_spread**2)))
        
        # Combine
        dist = dist_left + dist_right
        
        # Normalize so sum is 1 (Probability Mass Function representing ONE "parallel event" - 2 people)
        # Note: If we treat "1 parallel event" as "2 people crossing", the integral should sum to 1 event.
        # But later we multiply by "number of people". 
        # If 'people_per_year' counts INDIVIDUALS, then a parallel event of 2 people consumes 2 counts?
        # Let's clarify: The input is "people_per_year".
        # If 20% people walk in parallel, that means 0.2 * N people are involved in parallel walking.
        # That means (0.2 * N) / 2 "parallel groups" passed by.
        # The distribution should represent the spatial probability of *a person* in that mode.
        # So we normalize the sum to 1. This means "Given a person is walking in parallel mode, where are they likely to step?"
        
        total = np.sum(dist)
        if total > 0:
            dist = dist / total
            
        return dist

    def simulate_wear(self, years, people_per_year, 
                      parallel_ratio=0,
                      up_down_split=1,
                      traffic_dist_func_single=None, 
                      traffic_dist_func_parallel=None,
                      correction_func=None):
        """
        Simulates wear over a period.
        
        Args:
            years (float): Number of years.
            people_per_year (float): Number of people (footsteps) per year.
            parallel_ratio (float): Fraction of people walking in parallel (0.0 to 1.0).
            up_down_split (float): Fraction of people walking UP (0.0 to 1.0).
            traffic_dist_func_single (callable): Custom function for single traffic.
            traffic_dist_func_parallel (callable): Custom function for parallel traffic.
            correction_func (callable): Function returning correction coefficient.
        """
        if self.wear_kernel_up is None or self.wear_kernel_down is None:
            self.init_kernels()
            
        # 1. Determine Total Traffic Volume
        total_people = years * people_per_year
        
        people_up = total_people * up_down_split
        people_down = total_people * (1 - up_down_split)
        
        # Helper to process one direction
        def process_direction(n_people, kernel, direction_label):
            if n_people <= 0:
                return np.zeros_like(self.stair_grid)
                
            # Split into single and parallel groups
            people_p = n_people * parallel_ratio
            people_s = n_people * (1 - parallel_ratio)
            
            # Determine offset
            # Center is 200.
            # Up: Closer to outer edge (Nosing). +50mm (Shift to 250).
            # Down: -20mm (Shift to 180).
            
            y_offset_dir = 0
            if direction_label == 'up':
                y_offset_dir = 200
            elif direction_label == 'down':
                y_offset_dir = 0
            
            # Get Traffic Distributions
            if traffic_dist_func_single is None:
                prob_map_single = self.get_single_traffic_distribution(y_offset=y_offset_dir)
            else:
                prob_map_single = traffic_dist_func_single()
                
            if traffic_dist_func_parallel is None:
                prob_map_parallel = self.get_parallel_traffic_distribution(y_offset=y_offset_dir)
            else:
                prob_map_parallel = traffic_dist_func_parallel()
                
            # Calculate Step Counts Maps
            map_counts = (prob_map_single * people_s) + (prob_map_parallel * people_p)
            
            # Apply Correction
            if correction_func is not None:
                map_counts *= correction_func()
                
            # Convolve
            return convolve2d(map_counts, kernel, mode='same', boundary='fill', fillvalue=0)

        # Accumulate wear
        wear_up = process_direction(people_up, self.wear_kernel_up, 'up')
        wear_down = process_direction(people_down, self.wear_kernel_down, 'down')
        
        self.stair_grid += (wear_up + wear_down)

    def visualize(self, output_dir='output'):
        """
        Visualizes the stair wear simulation results.
        Saves separate PNG files for each plot to the output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.wear_kernel_up is None or self.wear_kernel_down is None:
            self.init_kernels()

        print(f"Saving visualizations to {os.path.abspath(output_dir)}...")

        # 1. Wear Depth Map (2D)
        plt.figure(figsize=(10, 5))
        # Origin 'upper' places index 0 at top. 
        # If Y=0 is Wall and Y=400 is Nosing, and we want Nosing (Outer Edge) at Screen (Bottom),
        # we want Y=400 at Bottom. 
        # With origin='upper', Y=0 is Top, Y=400 is Bottom. This matches "Outer edge facing screen".
        # cmap 'inferno': Black (Low/No Wear) -> Yellow (High/Deep Wear).
        # This matches 3D plot where Deep Wear (Low Z) is Yellow (via inferno_r on negative Z).
        plt.imshow(self.stair_grid, cmap='inferno', origin='upper', 
                   extent=[0, self.length, self.width, 0], aspect='equal')
        plt.title(f"Stair Wear Depth (Max: {np.max(self.stair_grid):.4f} mm)")
        plt.xlabel("Stair Length (mm)")
        plt.ylabel("Stair Tread Depth (mm) [Bottom is Outer Edge]")
        plt.colorbar(label='Wear Depth (mm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_wear_depth_map.png'))
        plt.close()
        
        # 2. Kernel Up
        plt.figure(figsize=(5, 8))
        plt.imshow(self.wear_kernel_up, cmap='inferno', origin='lower',
                   extent=[0, self.foot_width, 0, self.foot_length], aspect='equal')
        plt.title(f"UP Step Wear Kernel\n(Max: {np.max(self.wear_kernel_up):.2e} mm)")
        plt.xlabel("Foot Width (mm)")
        plt.ylabel("Foot Length (mm)")
        plt.colorbar(label='Wear Depth (mm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_kernel_up.png'))
        plt.close()
        
        # 3. Kernel Down
        plt.figure(figsize=(5, 8))
        plt.imshow(self.wear_kernel_down, cmap='inferno', origin='lower',
                   extent=[0, self.foot_width, 0, self.foot_length], aspect='equal')
        plt.title(f"DOWN Step Wear Kernel\n(Max: {np.max(self.wear_kernel_down):.2e} mm)")
        plt.xlabel("Foot Width (mm)")
        plt.ylabel("Foot Length (mm)")
        plt.colorbar(label='Wear Depth (mm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_kernel_down.png'))
        plt.close()
        
        # 4. Cross Sections
        # Length
        plt.figure(figsize=(10, 4))
        mid_y = self.ny // 2
        plt.plot(np.linspace(0, self.length, self.nx), self.stair_grid[mid_y, :], label='Mid-Depth Profile')
        plt.title("Cross Section: Wear along Length (Center)")
        plt.xlabel("Length (mm)")
        plt.ylabel("Wear Depth (mm)")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_profile_length.png'))
        plt.close()
        
        # Width
        plt.figure(figsize=(6, 4))
        mid_x = self.nx // 2
        plt.plot(np.linspace(0, self.width, self.ny), self.stair_grid[:, mid_x], color='orange', label='Mid-Length Profile')
        plt.title("Cross Section: Wear along Width (Center)")
        plt.xlabel("Width/Depth (mm)")
        plt.ylabel("Wear Depth (mm)")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5_profile_width.png'))
        plt.close()

        # 5. 3D Surface Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, self.length, self.nx)
        y = np.linspace(0, self.width, self.ny)
        X, Y = np.meshgrid(x, y)
        Z = -self.stair_grid # 负值表示磨损深度
        
        surf = ax.plot_surface(X, Y, Z, cmap='inferno_r', linewidth=0, antialiased=False)
        ax.set_title("3D Stair Surface Model (Wear Exaggerated)")
        ax.set_xlabel("Length (mm)")
        ax.set_ylabel("Depth (mm)")
        ax.set_zlabel("Height (mm)")
        
        # 设置3D图的长宽比
        ax.set_box_aspect((self.length, self.width, self.width / 3))
        
        # 调整视角：外缘（Nosing, Y=max）朝向屏幕
        # azim=90 表示从Y轴正向看去
        ax.view_init(elev=40, azim=110) 
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Surface Height (mm)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_surface_3d.png'))
        plt.close()
        
        print("Done.")

if __name__ == "__main__":
    print("Initializing Stair Wear MVP...")
    # Default: Stone (200 MPa), Rubber on Stone (k=1e-4), 80kg walker
    mvp = StairWearMVP(length=1000, width=400, resolution=10)
    
    print("Simulating Wear...")
    # Simulate 100 years, 500 people per day -> ~182,500 people/year
    # Total ~18 million steps
    years = 100
    people_per_day = 500
    mvp.simulate_wear(years=years, people_per_year=people_per_day * 365)
    
    print(f"Max Wear Depth: {np.max(mvp.stair_grid):.4f} mm")
    
    print("Visualizing...")
    mvp.visualize()
    print("Done.")
