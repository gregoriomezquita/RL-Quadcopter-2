import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
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
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        
        #self.action_repeat = 1

        #self.state_size = self.action_repeat * 6
        self.state_size = self.sim.pose.shape[0]+ self.sim.v.shape[0]
        self.action_low = 10
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        
        self.labels = ['time', 'x', 'y', 'z',
                        'phi', 'theta', 'psi', \
                        'x_velocity', 'y_velocity', 'z_velocity', \
                        'phi_velocity', 'theta_velocity', 'psi_velocity',\
                        'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', \
                        'reward']
          
        self.records= {x : [] for x in self.labels}

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #return np.sqrt( ( (self.sim.pose[:3] - self.target_pos)**2).sum() )
        '''
        reward= np.sqrt( ( (self.sim.pose[:3] - self.target_pos)**2).sum() )
        #reward-= (reward - (500*self.sim.time))
        reward= np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        #reward = 1.- (.3*(abs(self.sim.pose[:3] - self.target_pos)).sum() )
        '''
        norm= np.linalg.norm([self.sim.pose[:3] - self.target_pos])
        reward= -min(norm, 20)
        #reward = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.0)
        if self.sim.pose[2] >= self.target_pos[2]: reward+= 10
        if self.sim.pose[2] >= self.target_pos[2]+ 1: reward-= 10
        if self.sim.v[0] < 0.2: reward+= 10
        if self.sim.v[1] < 0.2: reward+= 10
        if self.sim.v[2] > 0.2: reward+= 10
        
        #if self.sim.time > self.sim.runtime: reward-= 10
        #if z >= target_z+ 0.5: reward+= 5.0
        #reward+= 0.5*self.sim.time
        #reward= -((self.target_pos- self.sim.pose[:3]).sum() - (0.7*self.steps))
        #reward= -((self.target_pos- self.sim.pose[:3]).sum() - (100*self.sim.time))

        #reward= -((self.target_pos[2]- self.sim.pose[2]).sum() - (100*self.sim.time))
        #reward= -(self.target_pos[2]- self.sim.pose[2])
        #reward = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.0) 
        return reward
        #return np.clip(reward, -1.0, 1.0)
        
        '''
        reward = 0
        reward -= np.linalg.norm([self.sim.pose[:3] - self.target_pos])
        reward += 10.0/np.abs(self.sim.pose[2] - self.target_pos[2])
        if self.sim.v[2] > 0.2: reward+=10
        return reward
        '''
        '''
        reward= 0
        reward-= np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        if self.sim.pose[2] >= self.target_pos[2]: reward+= 10
        #return np.clip(reward, -1.0, 1.0)
        return reward
        '''
        
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        '''
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        '''
        done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
        reward += self.get_reward()
        next_state = np.hstack((self.sim.pose, self.sim.v))
        # Compute rewards

        if self.sim.pose[2] >= self.target_pos[2]: done= True
        if self.sim.time > self.sim.runtime: done= True


        to_write = [self.sim.time] + list(self.sim.pose) + list(self.sim.v) + list(self.sim.angular_v) + list(rotor_speeds)+ [reward]
        for ii in range(len(self.labels)):
            self.records[self.labels[ii]].append(to_write[ii])
        #
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        state = np.hstack((self.sim.pose, self.sim.v))
        self.records = {x : [] for x in self.labels}
        return state

