#!/usr/bin/python

# Copyright 2006-2013 Dr. Marc Andreas Freese. All rights reserved. 
# marc@coppeliarobotics.com
# www.coppeliarobotics.com
# 
# -------------------------------------------------------------------
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# 
# You are free to use/modify/distribute this file for whatever purpose!
# -------------------------------------------------------------------
#
# This file was automatically created for V-REP release V3.0.3 on April 29th 2013

# Make sure to have the server side running in V-REP!
# Start the server from a child script with following command:
# simExtRemoteApiStart(19999) -- starts a remote API server service on port 19999

# after loading in V-rep the scene:  myYoubotScene.ttt
# start like:
# python myYouBotClient.py

import vrep
import sys
import numpy
import KTimage as KT
import math
import time


def img_to_numpy (res,img):
    #print "res = ", res
    colval = numpy.zeros((res[0]*res[1],3))
    i = 0
    for pix in range(res[0]*res[1]):
        for col in range(3):
            if  img[i] >= 0:
                colval[pix][col] = img[i]
            else:
                colval[pix][col] = img[i] + 256
            i += 1
    return colval


def img_binarise(colpic,r_min,r_max,g_min,g_max,b_min,b_max):
    #"""in a color image, search all pixels within a given color range, and export binary greyscale image in which the according pixels are white, all others black"""
    num_pixel = numpy.shape(colpic)[0]
    binpic = numpy.zeros(num_pixel)
    for pix in range(num_pixel):
        r = colpic[pix][0] / 255.0
        g = colpic[pix][1] / 255.0
        b = colpic[pix][2] / 255.0
        if  r >= r_min and r <= r_max and g >= g_min and g <= g_max and b >= b_min and b <= b_max:
            binpic[pix] = 1.0
    return binpic


def img_find_cm(binpic):
    #"""in a binary greyscale image, find the center of mass = (cm_x,cm_y) of those pixels that have values 1 (actually >0.5)"""
    if  numpy.max(binpic) < 0.5:
        return (-1,-1)
    else:
        cm_x, cm_y = 0.0, 0.0
        count = 0.0
        for x in range(numpy.shape(binpic)[0]):
            for y in range(numpy.shape(binpic)[1]):
                if  binpic[x][y] > 0.5:
                    cm_x += x
                    cm_y += y
                    count += 1.0
        cm_x = cm_x / count
        cm_y = cm_y / count
        cm_x = cm_x / numpy.shape(binpic)[0]
        cm_y = cm_y / numpy.shape(binpic)[1]
        return (cm_y,cm_y)


def printErr (err, textOK, textNotOK):
    if  err==vrep.simx_error_noerror:
        print(textOK)
    else:
        print("Error code = {}, {}".format(err, textNotOK))




print 'Program myYouBotClient.py started'
portID = 19997

# connect to V-rep
clientID=vrep.simxStart('127.0.0.1',portID,True,True,5000,5)
if  clientID!=-1:
    print('Connected to remote API server via portID {}'.format(portID))
else:
    print('Failed connecting to remote API server')
    vrep.simxFinish(clientID)
    exit(1)

#(re)start the simulation
vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)

# get handle/pointer to all objects in the V-rep scene
err,objs=vrep.simxGetObjects(clientID,vrep.sim_handle_all,vrep.simx_opmode_oneshot_wait)
printErr(err, 'Number of objects in the scene: {}'.format(len(objs)), 'Remote API function call returned with error code: {}'.format(err))

# show message in status bar below the V-REP main window
vrep.simxAddStatusbarMessage (clientID,"Connect from Python Client",vrep.simx_opmode_oneshot_wait)

# get handles for the 4 wheel motors
err,frontleftMotorHandle=vrep.simxGetObjectHandle(clientID,"rollingJoint_fl",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for front left motor: {}'.format(frontleftMotorHandle), 'Error by getting handle for front left motor: {}'.format(err))

err,rearleftMotorHandle=vrep.simxGetObjectHandle(clientID,"rollingJoint_rl",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for rear left motor: {}'.format(rearleftMotorHandle), 'Error by getting handle for rear left motor: {}'.format(err))

err,rearrightMotorHandle=vrep.simxGetObjectHandle(clientID,"rollingJoint_rr",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for rear right motor: {}'.format(rearrightMotorHandle), 'Error by getting handle for rear right motor: {}'.format(err))

err,frontrightMotorHandle=vrep.simxGetObjectHandle(clientID,"rollingJoint_fr",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for front right motor: {}'.format(frontrightMotorHandle), 'Error by getting handle for front right motor: {}'.format(err))

# get handles for the 5 arm joints
err,armJoint0Handle=vrep.simxGetObjectHandle(clientID,"youBotArmJoint0",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for arm joint 0: {}'.format(armJoint0Handle), 'Error by getting handle for arm joint 0: {}'.format(err))

err,armJoint1Handle=vrep.simxGetObjectHandle(clientID,"youBotArmJoint1",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for arm joint 1: {}'.format(armJoint1Handle), 'Error by getting handle for arm joint 1: {}'.format(err))

err,armJoint2Handle=vrep.simxGetObjectHandle(clientID,"youBotArmJoint2",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for arm joint 2: {}'.format(armJoint2Handle), 'Error by getting handle for arm joint 2: {}'.format(err))

err,armJoint3Handle=vrep.simxGetObjectHandle(clientID,"youBotArmJoint3",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for arm joint 3: {}'.format(armJoint3Handle), 'Error by getting handle for arm joint 3: {}'.format(err))

err,armJoint4Handle=vrep.simxGetObjectHandle(clientID,"youBotArmJoint4",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for arm joint 4: {}'.format(armJoint4Handle), 'Error by getting handle for arm joint 4: {}'.format(err))

# get handles for the 2 gripper joints
err,gripperJoint1Handle=vrep.simxGetObjectHandle(clientID,"youBotGripperJoint1",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for gripper joint 1: {}'.format(gripperJoint1Handle), 'Error by getting handle for gripper joint 1: {}'.format(err))

err,gripperJoint2Handle=vrep.simxGetObjectHandle(clientID,"youBotGripperJoint2",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for gripper joint 2: {}'.format(gripperJoint2Handle), 'Error by getting handle for gripper joint 2: {}'.format(err))

# get handles for the 2 attached cameras
err,visionSensor1Handle=vrep.simxGetObjectHandle(clientID,"Vision_sensor1",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for vision sensor 1: {}'.format(visionSensor1Handle), 'Error by getting handle for vision sensor 1: {}'.format(err))

err,visionSensor2Handle=vrep.simxGetObjectHandle(clientID,"Vision_sensor2",vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for vision sensor 2: {}'.format(visionSensor2Handle), 'Error by getting handle for vision sensor 2: {}'.format(err))

# set forces for the joint(motor)s
vrep.simxSetJointForce(clientID,armJoint0Handle,20,vrep.simx_opmode_oneshot)
vrep.simxSetJointForce(clientID,armJoint1Handle,20,vrep.simx_opmode_oneshot)
vrep.simxSetJointForce(clientID,armJoint2Handle,20,vrep.simx_opmode_oneshot)
vrep.simxSetJointForce(clientID,armJoint3Handle,20,vrep.simx_opmode_oneshot)
vrep.simxSetJointForce(clientID,armJoint4Handle,20,vrep.simx_opmode_oneshot)


counter = 0
# this loop won't end
while(err == vrep.simx_error_noerror or err == vrep.simx_error_novalue_flag):

    counter += 1
    print counter
    # move the robot somehow
    vrep.simxSetJointTargetVelocity(clientID,frontleftMotorHandle,0.5,vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(clientID,rearleftMotorHandle,0.5,vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(clientID,rearrightMotorHandle,-0.5,vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(clientID,frontrightMotorHandle,-0.5,vrep.simx_opmode_oneshot)

    # generate random target positions for the arm
    rand_base_rotate = numpy.random.uniform(-40,40)*math.pi/180
    rand_base_lower  = numpy.random.uniform(15,35) *math.pi/180
    rand_elbow_bend  = numpy.random.uniform(0,50) *math.pi/180
    rand_hand_bend   = numpy.random.uniform(10,90) *math.pi/180

    # instruct the arm to move (all joints at once)
    vrep.simxPauseCommunication(clientID,True)
    vrep.simxSetJointPosition(clientID,armJoint0Handle, rand_base_rotate,vrep.simx_opmode_streaming) # base - rotates the entire arm - [-40..40]
    vrep.simxSetJointPosition(clientID,armJoint1Handle, rand_base_lower ,vrep.simx_opmode_streaming) # base - lowers the arm - [5..35]
    vrep.simxSetJointPosition(clientID,armJoint2Handle, rand_elbow_bend ,vrep.simx_opmode_streaming) # elbow - bends the arm - [0..50]
    vrep.simxSetJointPosition(clientID,armJoint3Handle, rand_hand_bend  ,vrep.simx_opmode_streaming) # hand - bends the lower part of the arm - [0..90]
    vrep.simxSetJointPosition(clientID,armJoint4Handle, 0.0,vrep.simx_opmode_streaming) # wrist - turns the gripper
    vrep.simxPauseCommunication(clientID,False)

    # give the arm some time to move
    time.sleep(1)

    # verify that the arm has approximately reached its target position
    err0 = vrep.simxGetJointPosition(clientID,armJoint0Handle,vrep.simx_opmode_streaming)[1] - rand_base_rotate
    err1 = vrep.simxGetJointPosition(clientID,armJoint1Handle,vrep.simx_opmode_streaming)[1] - rand_base_lower
    err2 = vrep.simxGetJointPosition(clientID,armJoint2Handle,vrep.simx_opmode_streaming)[1] - rand_elbow_bend
    err3 = vrep.simxGetJointPosition(clientID,armJoint3Handle,vrep.simx_opmode_streaming)[1] - rand_hand_bend
    err4 = vrep.simxGetJointPosition(clientID,armJoint4Handle,vrep.simx_opmode_streaming)[1] - 0.0
    joint_pos_err = abs(err0) + abs(err1) + abs(err2) + abs(err3) + abs(err4)
    print "joint positioning error =", joint_pos_err

    # get image from vision sensor 1
    err,res,img = vrep.simxGetVisionSensorImage(clientID,visionSensor1Handle,0,vrep.simx_opmode_oneshot_wait)
    colval1 = img_to_numpy(res,img)
    # get the binary image marking the cyan ball in Youbot's gripper
    binval1 = img_binarise(colval1,0.0,0.15,0.7,0.99,0.9,1.0)
    # get the x- and y-coordinates of the cyan ball
    binval1 = numpy.reshape(binval1,(res[1],res[0]))
    pos_img1 = img_find_cm(binval1)
    print "pos_img1 =", pos_img1

    # get image from vision sensor 2
    err,res,img = vrep.simxGetVisionSensorImage(clientID,visionSensor2Handle,0,vrep.simx_opmode_oneshot_wait)
    colval2 = img_to_numpy(res,img)
    # get the binary image marking the cyan ball in Youbot's gripper
    binval2 = img_binarise(colval2,0.0,0.15,0.7,0.99,0.9,1.0)
    # get the x- and y-coordinates of the cyan ball
    binval2 = numpy.reshape(binval2,(res[1],res[0]))
    pos_img2 = img_find_cm(binval2)
    print "pos_img2 =", pos_img2

    # export images for look.tcl display
    if  counter % 5 == 0:
        KT.exporttiles (numpy.transpose(colval1), res[1], res[0], "/tmp/coco/obs_I_0.pgm", 3, 1)
        KT.exporttiles (binval1, res[1], res[0], "/tmp/coco/obs_J_0.pgm")
        KT.exporttiles (numpy.transpose(colval2), res[1], res[0], "/tmp/coco/obs_I_1.pgm", 3, 1)
        KT.exporttiles (binval2, res[1], res[0], "/tmp/coco/obs_J_1.pgm")

    # write the joint data and the ball-position data into one file (if joints are near target and if cyan ball is seen in both images)
    if  (joint_pos_err < 0.00001 and pos_img1[0] >= 0.0 and pos_img1[1] >= 0.0 and pos_img2[0] >= 0.0 and pos_img2[1] >= 0.0):
        fobj = open("outfile.dat", "a")
        fobj.write("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (rand_base_rotate, rand_base_lower, rand_elbow_bend, rand_hand_bend, pos_img1[0], pos_img1[1], pos_img2[0], pos_img2[1]))
        fobj.close()


print('Program ended')
exit(0)

