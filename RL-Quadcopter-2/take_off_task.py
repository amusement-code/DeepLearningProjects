import numpy as np
from physics_sim import PhysicsSim

class TakeOffTask(Task):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_velocities=None, 
        init_angle_velocities=None, runtime=5.):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation

        init_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        target_pos = np.array([0.0, 0.0, 10.0])
        self.done_reward = (abs(init_pose[:3] - self.target_pos)).sum()

        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, rotor_speeds, done):
        """
        Uses current pose of sim to return reward.

        the idea for this reward came from the walking robot in 
        lesson2: The RL Framework: The Problem - 10. Cumulative Reward (2:57 in the video)
        """
        sum_rotor_speed_coefficient = 1
        distance_from_target_coefficient = 1
        sum_rotor_speed_exponent_coefficient = 0.2

        # move fast
        sum_rotor_speed = 0
        for i in range(len(rotor_speeds)):
            sum_rotor_speed = min(rotor_speeds[i], self.action_high)

        # move to the right direction
        distance_from_target = (abs(self.sim.pose[:3] - self.target_pos)).sum()
        # distance_from_target = ((self.sim.pose[:3] - self.target_pos)**2).sum()

        # move smoothly
        sum_rotor_speed_exponent = 0
        for i in range(len(rotor_speeds)):
            sum_rotor_speed_exponent += rotor_speeds[i]
            # sum_rotor_speed_exponent += rotor_speeds[i]**2
        # sum_rotor_speed_exponent = sum_rotor_speed_exponent**2

        
        reward = sum_rotor_speed_coefficient * sum_rotor_speed - \
                    distance_from_target_coefficient * distance_from_target - \
                    sum_rotor_speed_exponent_coefficient * sum_rotor_speed_exponent

        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds, done) 
            pose_all.append(self.sim.pose)
        
        if done:
            reward += self.done_reward
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state