[[_TOC_]]

## Relevant links

### Python client library for ROS (reference)

- ```rospy```: http://docs.ros.org/en/melodic/api/rospy/html/rospy-module.html
- ```rospy.Service```: http://docs.ros.org/en/melodic/api/rospy/html/rospy.impl.tcpros_service.Service-class.html
- ```rospy.ServiceProxy```: http://docs.ros.org/en/melodic/api/rospy/html/rospy.impl.tcpros_service.ServiceProxy-class.html

### Common ROS Messages (Topics)

- Standard ROS Messages (**std_msgs**): http://wiki.ros.org/std_msgs 
- Geometric primitives (**geometry_msgs**): http://wiki.ros.org/geometry_msgs 
- Commonly used sensors (**sensor_msgs**): http://wiki.ros.org/sensor_msgs 

### Common ROS Messages (Services)

- Sending a signal to a ROS node (**std_srvs**): http://wiki.ros.org/std_srvs

## Create a ROS package for custom messages, services and actions 

Create package by replacing ```PACKAGE_NAME_msgs``` with the name of the package (e.g. *my_messages_msgs*)

```bash
catkin_create_pkg PACKAGE_NAME_msgs std_msgs geometry_msgs sensor_msgs message_generation message_runtime
```

Create a folder for the message definition according to the message type

- **msg**: describe the ROS message for topics
- **srv**: describe a service 
- **action**: describe a ROS action

```
ros_ws
  |-- build
  |-- devel
  |-- src
  |    |-- my_messages_msgs
  |    |    |-- msg
  |    |    |-- srv
  |    |    |-- action
```
