import numpy as np
from numba import jit
np.random.seed(1)


class rocket_engine():

    def __init__(self, dimensions = 3, temperature = 1e4, N = 1e+5, mass = 1.67e-27, length = 1e-6):
        self.k = 1.38064852e-23
        self.T = temperature
        self.N = N
        self.m = mass
        self.L = length 
        self.dim = dimensions
        self.sigma = np.sqrt((self.k*self.T)/self.m)
        self.x = self.position()
        self.v = self.velocities()
    
    def velocities(self):
        return np.random.normal(0,self.sigma, size=(int(self.N),self.dim))

    def position(self):
        return np.random.uniform(0,self.L,   size=(int(self.N),self.dim))


#Calculating the mean velocity
    def meanvel(self):
        self.v_s = 0
        for i in range(int(self.N)):
            self.v_s += np.sqrt(self.v[i,0]**2+self.v[i,1]**2+self.v[i,2]**2)
        return self.v_s/self.N

    def meankin(self):
        self.v_s = 0
        for i in range(int(self.N)):
            self.v_s += self.v[i,0]**2+self.v[i,1]**2+self.v[i,2]**2
        return 0.5*self.m *(self.v_s/self.N)



    def box_escape(self, steps = 1e4, t_end = 1e-9, dt = 1e-12):
        x, v = self.x,self.v
        position = self.position()
        velocity = self.velocities()
        exiting = 0.0
        exiting_velocities = 0.0 
        for t in range(int(steps)):
            x += v * dt  
            v_exiting = np.abs(v[:,2])
            collision_points = np.logical_or(np.less_equal(x, 0.), np.greater_equal(x, self.L))
            x_mask = np.logical_or(np.greater_equal(x[:,0], 0.25*self.L), np.less_equal(x[:,0], 0.75*self.L))
            y_mask = np.logical_and(np.greater_equal(x[:,0], 0.25*self.L), np.less_equal(x[:,0], 0.75*self.L))

            exit_points = np.logical_and(x_mask, y_mask)
            exit_points = np.logical_and(np.less_equal(x[:,2], 0), exit_points)
            exit_indices = np.where(exit_points == True)
            not_exit_indices = np.where(exit_points == True)
            v_exiting[not_exit_indices] = 0. 
            exiting_velocities += np.sum(v_exiting)

            collision_indices = np.where(collision_points == True)
            exiting_velocities += len(exit_indices[0])
            s_matrix = np.ones_like(x)
            s_matrix[collision_indices] = -1 
            s_matrix[:,2][exit_indices] = 1
            r_matrix = np.zeros_like(x) 
            x[:,2][exit_indices] += 0.99*self.L 
            v = np.multiply(v,s_matrix)
            particle_per_second =  exiting_velocities/t_end
        return exiting



if __name__ == "__main__":
    A = rocket_engine()
    result2 = A.box_escape()
    print(result2)
    

