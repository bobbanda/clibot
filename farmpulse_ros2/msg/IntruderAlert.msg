# IntruderAlert.msg
# High-level alert message when intruders are detected

# Header with timestamp
std_msgs/Header header

# Alert severity: INFO, WARNING, CRITICAL
string severity

# Number of intruders detected
int32 num_intruders

# Highest confidence score
float32 max_confidence

# Array of all detections
IntruderDetection[] detections

# Location description
string location

# Alert message
string message

# Suggested action
string action