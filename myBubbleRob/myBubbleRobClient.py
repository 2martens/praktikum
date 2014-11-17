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
# This file was automatically created for V-REP release V3.0.3 on April
# 29th 2013

# Make sure to have the server side running in V-REP!
# Start the server from a child script with following command:
# simExtRemoteApiStart(19999) -- starts a remote API server service on
# port 19999


import vrep
import sys
import time
import numpy
import KTimage as KT


def img_to_numpy(res, img):
    # print "res = ", res
    colval = numpy.zeros((res[0] * res[1], 3))
    i = 0
    for pix in range(res[0] * res[1]):
        for col in range(3):
            if img[i] >= 0:
                colval[pix][col] = img[i]
            else:
                colval[pix][col] = img[i] + 256
            i += 1
    return colval


def printErr(err, textOK, textNotOK):
    if err == vrep.simx_error_noerror:
        print(textOK)
    else:
        print("Error code = {}, {}".format(err, textNotOK))


print('Program myBubbleRobClient.py started')
portID = 19997
clientID = vrep.simxStart('127.0.0.1', portID, True, True, 5000, 5)
if clientID != -1:
    print('Connected to remote API server via portID {}'.format(portID))
else:
    print('Failed connecting to remote API server')
    vrep.simxFinish(clientID)
    exit(1)

# show message in status bar below the V-REP main window
vrep.simxAddStatusbarMessage(
    clientID, "Connect from Python Client", vrep.simx_opmode_oneshot_wait)

# get handles for objects and robot parts
err, objs = vrep.simxGetObjects(
    clientID, vrep.sim_handle_all, vrep.simx_opmode_oneshot_wait)
printErr(err, 'Number of objects in the scene: {}'.format(len(objs)),
         'Remote API function simxGetObjects returned error: {}'.format(err))

err, leftMotorHandle = vrep.simxGetObjectHandle(
    clientID, "remoteApiControlledBubbleRobLeftMotor", vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for left motor: {}'.format(leftMotorHandle),
         'Error by getting handle for left motor: {}'.format(err))

err, rightMotorHandle = vrep.simxGetObjectHandle(
    clientID, "remoteApiControlledBubbleRobRightMotor", vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for right motor: {}'.format(
    rightMotorHandle), 'Error by getting handle for right motor: {}'.format(err))

err, visionSensorHandle = vrep.simxGetObjectHandle(
    clientID, "Vision_sensor", vrep.simx_opmode_oneshot_wait)
printErr(err, 'Get handle for vision sensor: {}'.format(
    visionSensorHandle), 'Error by getting handle for vision sensor: {}'.format(err))

counter = 0
while (True):

    # move the robot somehow
    vrep.simxSetJointTargetVelocity(
        clientID, leftMotorHandle, 3.1415 * 0.25, vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(
        clientID, rightMotorHandle, 3.1415 * 0.1, vrep.simx_opmode_oneshot)

    # get vision sensor image
    err, res, img = vrep.simxGetVisionSensorImage(
        clientID, visionSensorHandle, 0, vrep.simx_opmode_oneshot_wait)
    colval = img_to_numpy(res, img)
    if counter % 10 == 0:
        KT.exporttiles(
            numpy.transpose(colval), res[1], res[0], "/tmp/coco/obs_I_0.pgm", 3, 1)

    # this is only for the err return value - some other functions don't
    # return a reliable err code!
    err, objs = vrep.simxGetObjects(
        clientID, vrep.sim_handle_all, vrep.simx_opmode_oneshot_wait)
    if err != vrep.simx_error_noerror:
        print("Error in loop: simxGetObjects can't get handles anymore!")
        exit(2)

    # initialize to start
    if counter % 100 == 0:
        # stop simulation
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
        # wait a sec .. else the restart doesn't work
        time.sleep(1)
        # restart the simulation
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)

    counter += 1

print('Program exiting loop due to err = {}'.format(err))
vrep.simxFinish(clientID)

print('Program ended')
vrep.simxFinish(clientID)
exit(0)
