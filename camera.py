import numpy as np
import cv2 as cv
from math import *
import sys
from math import *
import time
import numpy as np
from NatNetClient import NatNetClient
from util import quaternion_to_euler_angle_vectorized1
import socket
import time
import matplotlib.pyplot as plt

def avg(lst):
    return sum(lst) / len(lst)

def avg2Pt(lst):
    Xsum = 0
    Ysum = 0
    for pt in lst:
        Xsum += pt[0]
        Ysum += pt[1]
    
    X = Xsum / len(lst)
    Y = Ysum / len(lst)

    return X, Y

# u to theta equation
def uTheta(u):
    return (-5e-11*(u**4)) + (6e-08*(u**3)) - (2e-05*(u**2)) - (0.0015*u) + (0.6468)

# sb to D conversion
def sbD(sb):
    i = 1/sb
    return (1712.5*i) + (5.6904)

def compute_object_position(x, y, alpha, D, theta):

	#Global coords of robot considering angle
    alpha = radians(alpha)

	#Rotation matrix
    R = np.array([[cos(theta), -sin(theta), x],

          	[sin(theta), cos(theta), y],

          	[0, 0, 1]])

	#local coord matrix of target
    target_local = np.array([[D * cos(alpha)], [D * sin(alpha)], [1]])

	#relate target using rotation
    target_global = np.matmul(R, target_local)

    #Duck coords actual
    Duck_x_global = target_global[0][0]
    Duck_y_global = target_global[1][0]
    return Duck_x_global, Duck_y_global

# Camera setup
cap = cv.VideoCapture('http://192.168.0.209:3005/stream.mjpg')

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Robot connection setup
IP_ADDRESS = '192.168.0.209'

# Connect to the robot
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((IP_ADDRESS, 5000))
print('Connected')

positions = {}
rotations = {}
thetaArr = []
dArr = []
duckArr = []
duckFinalArr = []
pt = [0, 0]


# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame(robot_id, position, rotation_quaternion):
    # Position and rotation received
    positions[robot_id] = position
    # The rotation is in quaternion. We need to convert it to euler angles

    rotx, roty, rotz = quaternion_to_euler_angle_vectorized1(rotation_quaternion)

    rotations[robot_id] = rotz


if __name__ == "__main__":
    import socket
    ## getting the hostname by socket.gethostname() method
    hostname = socket.gethostname()
    ## getting the IP address using socket.gethostbyname() method
    ip_address = socket.gethostbyname(hostname)
    print(ip_address)
    
    clientAddress =  "192.168.0.6"
    optitrackServerAddress = "192.168.0.4"
    robot_id = 209

    # This will create a new NatNet client
    streaming_client = NatNetClient()
    streaming_client.set_client_address(clientAddress)
    streaming_client.set_server_address(optitrackServerAddress)
    streaming_client.set_use_multicast(True)
    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()

while is_running:
    if robot_id in positions:
        try:
            command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(2000, 2000, 1500, 1500)
            s.send(command.encode('utf-8'))

            ret, frame = cap.read()
            
            # It converts the BGR color space of image to HSV color space
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            # Threshold of blue in HSV space
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # preparing the mask to overlay
            mask = cv.inRange(hsv, lower_yellow, upper_yellow)
            
            result = cv.bitwise_and(frame, frame, mask = mask)
            gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            gray = cv.medianBlur(gray, 5)
                
            rows = gray.shape[0]
            #frame2 = picam2.capture_file("blob.png")
            #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            params = cv.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 3000
            params.maxArea = 100000
            params.filterByColor = False
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            
            
            # Set up the detector with default parameters.
            detector = cv.SimpleBlobDetector_create(params)
            
            # Detect blobs.
            keypoints = detector.detect(gray)
            if len(keypoints) >= 1:
                max = keypoints[0]
                for keypoint in keypoints:
                    if keypoint.size > max.size:
                        max = keypoint
                theta = uTheta(max.pt[0])
                thetaArr.append(theta)
                d = sbD(max.size * 100)
                dArr.append(d)
                duckPos = compute_object_position(positions[robot_id][0], radians(positions[robot_id][1]), rotations[robot_id], d, theta)
                print('len duckArr: ', len(duckArr))
                if len(duckArr) == 0:
                    duckArr.append(duckPos)
                    pt = avg2Pt(duckArr)
                else:
                    errx = abs(pt[0] - duckPos[0])
                    erry = abs(pt[1] - duckPos[1])
                    print('sqerr: ', sqrt(errx**2 + erry**2))
                    if sqrt(errx**2 + erry**2) > 3:
                        duckFinalArr.append(pt)
                        duckArr = []
                        if len(duckFinalArr) == 5:
                            print(duckFinalArr)
                            break
                    else:
                        duckArr.append(duckPos)
                        pt = avg2Pt(duckArr)
                print('Robot Position: ', positions[robot_id][0], positions[robot_id][1])
                print('Calculated Current Point: ', duckPos)
                print('Average Current Point: ', pt)
            frame_with_keypoints = cv.drawKeypoints(result, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            #im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Show keypoints
            cv.imshow("Keypoints", frame_with_keypoints)
            
            #cv.waitKey(0)
            #print('cv.waitKey(0)')
            
            # Display the resulting frame
            #cv2.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        except KeyboardInterrupt:
            break

# When everything done, release the capture
command = 'CMD_MOTOR#00#00#00#00\n'
s.send(command.encode('utf-8'))
s.shutdown(2)
s.close()
streaming_client.shutdown()
cv.destroyAllWindows()
print('cv.destroyAllWindows()')

plt.grid()
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(duckFinalArr[:][0], duckFinalArr[:][1], label="Duck Position")
leg = plt.legend(loc='upper center')
plt.show()