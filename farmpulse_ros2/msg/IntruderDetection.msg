# IntruderDetection.msg
# Message type for individual intruder detection

# Header with timestamp
std_msgs/Header header

# Bounding box coordinates (pixels)
int32 x1
int32 y1
int32 x2
int32 y2

# Detection confidence (0.0-1.0)
float32 confidence

# Center point of detection
geometry_msgs/Point center

# Detection ID for tracking
string detection_id

# Zone/area where detected (optional)
string zone