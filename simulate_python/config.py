ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 0 # Domain id
INTERFACE = "wlp3s0" # Interface 

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

ENABLE_DEPTH_VISUALIZATION = True # Show a live depth window for the configured camera
DEPTH_CAMERA_NAME = "base_camera" # Camera used for depth visualization
DEPTH_VISUALIZATION_DT = 0.1 # Refresh period of the depth window
DEPTH_AUTO_RANGE = True # Auto-scale depth image using valid pixels in the current frame
DEPTH_RANGE = (0.2, 3.0) # Fixed depth range in meters when DEPTH_AUTO_RANGE is False

SIMULATE_DT = 0.002  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer
