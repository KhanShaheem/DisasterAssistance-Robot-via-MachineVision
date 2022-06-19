#!/usr/bin/python2

# include packages
import math
import rospy            # client library for ROS 1 (http://wiki.ros.org/rospy)

# import the custom service definitions
from custom_msgs.srv import RadiansToDegrees, RadiansToDegreesRequest, RadiansToDegreesResponse

CONVERSION_FACTOR_RADIANS_TO_DEGREES = 180 / math.pi     # conversion factor (radians -> degrees)


# service callback
def process_client_request(request):
    rospy.loginfo("Request from client: %4.2f", request.angle_radians)

    # create a response object
    response = RadiansToDegreesResponse()

    # perform sanity check (allow only positive real numbers)
    if(request.angle_radians < 0):
        response.angle_degrees = 0  # default error value
        response.success = False
    else:
        response.angle_degrees = CONVERSION_FACTOR_RADIANS_TO_DEGREES * request.angle_radians
        response.success = True

    #Return the response message.
    return response

def server():
    # register client node with the master under the specified name
    rospy.init_node("unit_converter_server", anonymous=False)

    # create a ROS service object
    service = rospy.Service('radians_to_degrees', RadiansToDegrees, process_client_request)

    # blocks program until ROS node is shutdown
    rospy.loginfo("Service is now available")
    rospy.spin()


# entry point to the program
if __name__ == "__main__":
    try:
        server()
    except rospy.ROSInterruptException:
        pass
