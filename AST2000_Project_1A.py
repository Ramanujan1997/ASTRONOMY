import numpy as np
from numba import jit
import sys 
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
#from ast2000tools.solar_system import SolarSystem 


np.random.seed(1)
#system = SolarSystem(seed)

def escape_velocity(M, R):
    v_escape = np.sqrt((2*G*M)/R)
    return v_escape 

    



class rocket_engine():

    def __init__(self, dimensions = 3, temperature = 10000, N = 1E5, mass = 1.67e-27, length = 1e-6):
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
        return self.v_s

    def meankin(self):
        m = self.m
        vel = 0
        for i in range(int(self.N)):
            vel += self.v[i,0]**2 + self.v[i,1]**2 + self.v[i,2]**2
        return 0.5 * m * vel

    def test_mean(self):
        """
        making a test function that runs meankin() and meanvel() and checks the
        computed velocity and kinetic energy and the relative error between them
        anything below 1% is perfectly acceptable
        """
        m = self.m
        analytical_mean = 1.5*self.T*self.k
        computed_mean   = 0 
        for j in self.v:
            computed_mean += self.meankin() 
            computed_mean  = computed_mean/self.N
            relative_error =    abs(analytical_mean - computed_mean)/analytical_mean
            print("----------Kinetic energy----------")
            print("{:<20}{:g}".format("Computed mean:", computed_mean))
            print("{:<20}{:g}".format("Analytical mean:", analytical_mean))
            print("{:<20}{:.2f}%".format("Relative error:", relative_error * 100))
            print("-----------------------------")
            break
        assert relative_error < 0.002, "the mean kinetic energy is off"

        print("----------Velocity----------")


        analytical_vel = np.sqrt(8*self.k*self.T/(np.pi*m))
        computed_vel   = 0
        for i in self.v: 
            computed_vel += self.meanvel()
            computed_vel = computed_vel/self.N 
            relative_error = abs(analytical_vel - computed_vel)/analytical_vel
            print("{:<20}{:g}".format("Computed velocity:", computed_vel))
            print("{:<20}{:g}".format("Analytical velocity:", analytical_vel))
            print("{:<20}{:.2f}%".format("Relative error:", relative_error *100))
            print("-----------------------------")
            break
        assert relative_error < 0.02, "the mean velocity is off"
    

    def box_escape(self, steps = 1e4, t_end = 1e-9, dt = 1e-12):
        """
        Checking how much of the particles actually escape the rocket
        steps: 
        t_end:
        dt: 

        """

        x, v = self.x,self.v
        exiting = 0.0
        exiting_velocities = 0.0 
        for t in range(int(t_end/dt)):
            x += v * dt  
            v_exiting = np.abs(v[:,2])
            collision_points = np.logical_or(np.less_equal(x, 0.), np.greater_equal(x, self.L))
            x_mask = np.logical_or(np.greater_equal(x[:,0], 0.25*self.L), np.less_equal(x[:,0], 0.75*self.L))
            y_mask = np.logical_and(np.greater_equal(x[:,0], 0.25*self.L), np.less_equal(x[:,0], 0.75*self.L))

            exit_points = np.logical_and(x_mask, y_mask)
            exit_points = np.logical_and(np.less_equal(x[:,2], 0), exit_points)
            exit_indices = np.where(exit_points == True)
            not_exit_indices = np.where(exit_points == False)
            v_exiting[not_exit_indices] = 0. 
            exiting_velocities += np.sum(v_exiting)

            collision_indices = np.where(collision_points == True)
            exiting += len(exit_indices[0])
            s_matrix = np.ones_like(x)
            s_matrix[collision_indices] = -1 
            s_matrix[:,2][exit_indices] = 1
            r_matrix = np.zeros_like(x) 
            x[:,2][exit_indices] += 0.99*self.L 
            v = np.multiply(v,s_matrix)
            particle_per_second =  exiting_velocities/t_end
        return exiting_velocities, exiting, particle_per_second

    def maxwell(self, x):
        sigma = np.sqrt(self.k * self.T/self.m)
        exponent = -x**2/(2*sigma**2)
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(exponent)


x1 = np.linspace(-25000, 25000, 51)
x_label = ["v_x", "v_y", "v_z"]
gas = rocket_engine()
for i, label in enumerate(x_label):
    plt.style.use("classic")
    plt.grid()
    plt.hist(gas.v[:,i], bins=31, density = True, histtype = "step")
    plt.plot(x1, gas.maxwell(x1), "r-")
    plt.show()


if __name__ == "__main__":
    A = rocket_engine()
    #result2 = A.box_escape()
    result3 = A.test_mean()

    

