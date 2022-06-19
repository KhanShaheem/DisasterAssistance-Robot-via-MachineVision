#!/usr/bin/python2

# include packages
import rospy            # client library for ROS 1 (http://wiki.ros.org/rospy)

# import the custom service definitions
from custom_msgs.srv import RadiansToDegrees, RadiansToDegreesRequest, RadiansToDegreesResponse


# service callback
def request_service(value):
    # wait for the service to become available
    rospy.loginfo("Waiting for the service...")
    rospy.wait_for_service('radians_to_degrees')

    # create a request object
    rospy.loginfo("Requesting conversion of %4.2f radians to degrees...", value)
    request = RadiansToDegreesRequest()
    request.angle_radians = value
    
    try:
        # create a service proxy
        radians_to_degrees = rospy.ServiceProxy('radians_to_degrees', RadiansToDegrees)

        # call the service with the given request
        response = radians_to_degrees(request)

        # return the response to the calling function
        return response

    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", str(e))
        
        # handle a service error by returning a failed response 
        response = RadiansToDegreesResponse()
        response.angle_degrees = 0  # default error value
        response.success = False

        return response

def client():
    # register client node with the master under the specified name
    rospy.init_node("unit_converter_client", anonymous=False)

    value_in_radians = 1.5708

    # call the service
    response = request_service(value_in_radians)

    # process the response and display log messages accordingly
    if(not response.success):
        rospy.logerr("Conversion unsuccessful! There was an error on the server or the request did not comply with the format (positive real number)")
    else:
        rospy.loginfo("Conversion successful: %4.2f radians = %4.2f degrees", value_in_radians, response.angle_degrees)


# entry point to the program
if __name__ == "__main__":
    try:
        client()
    except rospy.ROSInterruptException:
        pass
