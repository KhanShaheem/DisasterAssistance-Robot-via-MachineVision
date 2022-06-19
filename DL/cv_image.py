#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import face_recognition
import cv2
import numpy as np

julen_1_image = face_recognition.load_image_file("Images/Julen_1.jpeg")
julen_face_encoding = face_recognition.face_encodings(julen_1_image)[0]

ander_image = face_recognition.load_image_file("Images/Ander.jpeg")
ander_face_encoding = face_recognition.face_encodings(ander_image)[0]

alejandro_image = face_recognition.load_image_file("Images/Alejandro.jpeg")
alejandro_face_encoding = face_recognition.face_encodings(alejandro_image)[0]

shaheem_image = face_recognition.load_image_file("Images/Shaheem.jpeg")
shaheem_face_encoding = face_recognition.face_encodings(shaheem_image)[0]

jon_image = face_recognition.load_image_file("Images/Jon.jpeg")
jon_face_encoding = face_recognition.face_encodings(jon_image)[0]

jonjavier_image = face_recognition.load_image_file("Images/JonJavier.jpeg")
jonjavier_face_encoding = face_recognition.face_encodings(jonjavier_image)[0]
# Create arrays of known face encodings and their names
known_face_encodings = [
    julen_face_encoding,
    ander_face_encoding,
    alejandro_face_encoding,
    shaheem_face_encoding,
    jon_face_encoding,
    jonjavier_face_encoding
]
known_face_names = [
    "Julen",
    "Ander",
    "Alejandro",
    "Shaheem",
    "Jon",
    "JonJavier"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eye_closed_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')


class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    #(rows,cols,channels) = cv_image.shape
    #if cols > 60 and rows > 60 :
      #cv2.circle(cv_image, (50,50), 10, 255)

    

    try:
      frame = self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)
    
        # Grab a single cv_image of video
    #ret, cv_image = video_capture.read()
     
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Resize cv_image of video to 1/4 size for faster face recognition processing
    small_cv_image = cv2.resize(cv_image, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_cv_image = small_cv_image[:, :, ::-1]

    # Only process every other cv_image of video to save time
    process_this_cv_image = True
    if process_this_cv_image:
        # Find all the faces and face encodings in the current cv_image of video
        face_locations = face_recognition.face_locations(rgb_small_cv_image)
        face_encodings = face_recognition.face_encodings(rgb_small_cv_image, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_cv_image = not process_this_cv_image

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the cv_image we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)
        roi_gray = gray[top:bottom, left:right]
        roi_gray_2 = gray[top:int((bottom+top)/2), left:right]
        roi_color = cv_image[top:bottom, left:right]
        # Draw a label with a name below the face
        cv2.rectangle(cv_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(cv_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        eyes = eye_cascade.detectMultiScale(roi_gray_2, 1.05, 2, minSize = (int(abs(right-left)/3.5), int(abs(top-bottom)/3.5)), maxSize = (int(abs(right-left)), int(abs(top-bottom))))
        eye_closed = eye_closed_cascade.detectMultiScale(roi_gray_2, 1.75, 2, minSize = (int(abs(right-left)/6), int(abs(top-bottom)/6)))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_closed = np.asarray(eye_closed)
            #cv2.circle(cv_image, (left, int((bottom+top)/2)), radius = 5, color = (255, 255, 0), thickness = -1)
            #cv2.circle(cv_image, (left, int(top)), radius = 5, color = (255, 255, 0), thickness = -1)
            if eye_closed.size == 0:
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(cv_image, 'Closed eyes', (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
            else:
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(cv_image, 'Opened eyes', (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)






def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()






if __name__ == '__main__':
    main(sys.argv)
    
