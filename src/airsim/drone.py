'''
this module will connect to the device and send the object to the server
'''
import airsim
import numpy as np
import time
import math


def transform_input(responses):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    from PIL import Image
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L'))
    print(im_final)
    print(im_final.shape)
    return im_final



class Device:

    def __init__(self):
        '''
        constructor
        initialize the basic variable and connection
        '''
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.initX = -.55265
        self.initY = -31.9786
        self.initZ = -1.0225
        self.action_space = 7
        pass

    def takeOff(self):
        '''
        function will be used to take off the drone
        :return: None
        '''
        self.client.takeoffAsync().join()
        # self.client.moveToPositionAsync(self.initX,
        #                                 self.initY,
        #                                 self.initZ,
        #                                 1).join()
        # self.client.moveByVelocityAsync(1,
        #                                 -0.67,
        #                                 -0.8,
        #                                 2).join()


    def current_state(self):
        '''
        this function give the current state of the drone
        :return:
        '''
        resp = self.client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)])
        current_state = transform_input(resp)
        return current_state

    def intepreteAction(self, action):
        '''
        this will convert action with scalling factor
        :param action: send by the Brain
        :return: convert into drone acceptable action
        '''
        scaling_factor = 0.25
        if action == 0:
            quad_offset = (0, 0, 0)
        elif action == 1:
            quad_offset = (scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, 0, scaling_factor)
        elif action == 4:
            quad_offset = (-scaling_factor, 0, 0)
        elif action == 5:
            quad_offset = (0, -scaling_factor, 0)
        elif action == 6:
            quad_offset = (0, 0, -scaling_factor)
        return quad_offset

    def compute_reward(self, quad_state, quad_vel, collision_info):
        '''
        this is the helping fuction will be used for computing reward
        :param quad_state: current state of the quadqwaptor
        :param quad_vel: velocity of the quadqwaptor
        :param collision_info: if there is any collution it will send
        :return: return the reward
        '''
        thresh_dist = 7
        beta = 1

        z = -10
        pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]),
               np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]),
               np.array([541.3474, 143.6714, -32.07256])]

        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))

        if collision_info.has_collided:
            reward = -100
        else:
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1]))) / np.linalg.norm(
                    pts[i] - pts[i + 1]))

            if dist > thresh_dist:
                reward = -10
            else:
                reward_dist = (math.exp(-beta * dist) - 0.5)
                reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
                reward = reward_dist + reward_speed

        return reward

    def moveDrone(self, action):
        '''
        this will take an action and perform on emulator.
        then it will calculate the reward based on collision information and velocity
        :param action: take and action.
        :return: reward
        '''
        quad_offset = self.intepreteAction(action)
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(quad_vel.x_val + quad_offset[0], quad_vel.y_val + quad_offset[1],
                                   quad_vel.z_val + quad_offset[2], 5).join()
        time.sleep(0.5)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        collision_info = self.client.simGetCollisionInfo()
        reward = self.compute_reward(quad_state, quad_vel, collision_info)
        return reward

    def step(self, action):
        reward =self.moveDrone(action)
        state = self.current_state()
        done = self.isDone(state, reward)
        return state, reward, done, "drone"
    def isDone(self, state, reward):
        done = 0
        if reward < -1:
            done = 1
        return done
    def reset(self):
        self.client.reset()
        return self.current_state()

