# Terrain Generation Tool
## Usage
1. First, install dependencies:
```bash
pip3 install noise opencv-python numpy 
```
2. Open `terrain_generator.py` and modify the initial configuration at the beginning. Here, we will use the Go2 robot as an example:
```python
# Robot directory
ROBOT = "go2"
# Input scene file
INPUT_SCENE_PATH = "./scene.xml"
# Output
OUTPUT_SCENE_PATH = "../unitree_robots/" + ROBOT + "/scene_terrain.xml"
```
3. Run:
```bash
cd terrain_tool
python3 ./terrain_generator.py
```
If you want random boxes with guaranteed non-overlap, run:
```bash
python3 ./generate_random_boxes.py --count 120 --seed 24
```
The program will output the terrain scene file to `/unitree_robots/go2/scene_terrain.xml`. Then, you can modify the simulator configuration file `simulate/config.yaml` and set the scene to the newly generated `scene_terrain.xml`:
```yaml
robot_scene: "scene_terrain.xml"
```
If you are using a Python-based simulator, modify `simulate_python/config.py`:
```python
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene_terrain.xml" 
```
After that, run the unitree_mujoco simulator, and you can see the generated terrain.

## Export Isaac Lab Height Array To MuJoCo Hfield

If you already have a terrain height array exported from Isaac Lab (`.npy` or `.npz`),
you can convert it directly to a MuJoCo-compatible 16-bit PNG and MJCF parameters:

```bash
cd terrain_tool
python3 export_heightfield_from_array.py \
	--input ./lab_height.npy \
	--output-png ../unitree_robots/go2/lab_hfield_16u.png \
	--output-meta ../unitree_robots/go2/lab_hfield_meta.json \
	--cell-size 0.05 \
	--negative-depth 0.0
```

The script will:
- write a 16-bit height image (`--output-png`)
- write metadata + recommended MJCF snippet (`--output-meta`)
- print `<hfield>` and `<geom type="hfield">` XML snippet for direct copy

For `.npz` input with multiple arrays, choose one key:

```bash
python3 export_heightfield_from_array.py \
	--input ./lab_height.npz \
	--npz-key heights \
	--output-png ../unitree_robots/go2/lab_hfield_16u.png \
	--output-meta ../unitree_robots/go2/lab_hfield_meta.json \
	--cell-size 0.05
```

Tips:
- keep 16-bit PNG to reduce stair-step artifacts
- ensure `cell-size` matches your Lab terrain grid resolution (meters)
- use `--flip-y` if the map appears mirrored in MuJoCo

# Function Explanation
Users can utilize `terrain_generator.py` to add the desired terrain. Below is an explanation of the functions.
##### 1. `AddBox`
Add a cube, parameters:
```python
position=[1.0, 0.0, 0.0] # Center position
euler=[0.0, 0.0, 0.0] # Orientation
size=[0.1, 0.1, 0.1] # Size, length x width x height
``` 
##### 2. `AddGeometry`
Add a geometry, parameters:
```python
position=[1.0, 0.0, 0.0] # Center position
euler=[0.0, 0.0, 0.0] # Orientation
size=[0.1, 0.1, 0.1] # Size, some geometries only require the first two parameters
geo_type="cylinder" # Geometry type, supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
``` 
##### 3. `AddStairs`
Add stairs, parameters:
```python
init_pos=[1.0, 0.0, 0.0] # Position of the stair near the ground
yaw=0.0 # Stair orientation
width=0.2 # Stair width
height=0.15 # Stair height
length=1.5 # Stair length
stair_nums=10 # Number of stairs
```
##### 4. `AddSuspendStairs`
Add floating stairs, parameters:
```python
init_pos=[1.0, 0.0, 0.0] # Position of the stair near the ground
yaw=0.0 # Stair orientation
width=0.2 # Stair width
height=0.15 # Stair height
length=1.5 # Stair length
gap=0.1 # Floating gap
stair_nums=10 # Number of stairs
```
##### 5. `AddRoughGround`
Add rough terrain by randomly arranging cubes, parameters:
```python
init_pos=[1.0, 0.0, 0.0] # Position of the first cube
euler=[0.0, -0.0, 0.0], # Terrain orientation relative to the world
nums=[10, 10], # Number of cubes in x and y directions
box_size=[0.5, 0.5, 0.5], # Cube size
box_euler=[0.0, 0.0, 0.0], # Cube orientation
separation=[0.2, 0.2], # Cube separation in x and y directions
box_size_rand=[0.05, 0.05, 0.05], # Random increment of cube size
box_euler_rand=[0.2, 0.2, 0.2], # Random increment of cube orientation
separation_rand=[0.05, 0.05] # Random increment of cube separation
```

##### 6.`AddPerlinHeighField`
Generate terrain based on Perlin noise, parameters:
```python
position=[1.0, 0.0, 0.0],  # Terrain center position
euler=[0.0, 0.0, 0.0],  # Terrain orientation relative to the world
size=[1.0, 1.0],  # Terrain length and width
height_scale=0.2,  # Maximum terrain height
negative_height=0.2,  # Negative height in the z-axis direction
image_width=128,  # Terrain height map image pixel size
image_height=128,
smoothness=100.0,  # Noise smoothness
perlin_octaves=6,  # Perlin noise parameters
perlin_persistence=0.5,
perlin_lacunarity=2.0,
output_heightmap_image="height_field.png"  # Output height map image name
```

##### 7. `AddHeighFieldFromImage`
Generate terrain based on a given image, parameters:
```python
position=[1.0, 0.0, 0.0] # Terrain center position
euler=[0.0, 0.0, 0.0],  # Terrain orientation relative to the world
size=[2.0, 1.6],  # Terrain length and width
height_scale=0.02,  # Maximum terrain height
negative_height=0.1,  # Negative height in the z-axis direction
input_image_path="./unitree_robot.jpeg" # Input image path
output_heightmap_image="height_field.png", # Output height map image name
image_scale=[1.0, 1.0],  # Image scaling factors
invert_grayscale=False # Invert pixel
```
