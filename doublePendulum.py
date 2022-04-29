import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio

from utils.shared.integrators import myRK4 

def doublePendulumODE(t,y,p):
    # p = [m1,m2,l1,l2,g]
    m1 = p[0]
    m2 = p[1]
    l1 = p[2]
    l2 = p[3]
    g = p[4]

    theta1 = y[0]
    theta2 = y[2]

    dydt1 = y[1] #theta1dot

    dydt2 = (m2*g*np.sin(theta2)*np.cos(theta1-theta2) - m2*np.sin(theta1-theta2)*(l1*y[1]**2*np.cos(theta1-theta2) + l2*y[3]**2)
            - (m1+m2)*g*np.sin(theta1)) / l1 / (m1 + m2*np.sin(theta1-theta2)**2) #theta1ddot

    dydt3 = y[3] #theta2dot

    dydt4 = ((m1+m2)*(l1*y[1]**2*np.sin(theta1-theta2) - g*np.sin(theta2) + g*np.sin(theta1)*np.cos(theta1-theta2))
            + m2*l2*y[3]**2*np.sin(theta1-theta2)*np.cos(theta1-theta2)) / l2 / (m1 + m2*np.sin(theta1-theta2)**2) #theta2ddot

    return np.array((dydt1,dydt2,dydt3,dydt4))

if __name__ == "__main__":
    
    m1 = 1
    m2 = m1
    l1 = 1
    l2 = l1
    g = 9.81
    parameters = np.array([m1,m2,l1,l2,g])

    theta1_0 = np.radians(90)
    theta2_0 = np.radians(136)
    thetadot1_0 = np.radians(0)
    thetadot2_0 = np.radians(0)

    initialConditions = np.array([theta1_0,thetadot1_0,theta2_0,thetadot2_0],dtype=np.float64)
    
    tStart = 0
    tEnd = 10
    tSpan = np.array([tStart,tEnd])
    dt = 0.01
    tSpanExplicit = np.linspace(tStart,tEnd,int(tEnd / dt))

    start = time.time()
    sol = myRK4(doublePendulumODE,initialConditions,tSpanExplicit,parameters)
    end = time.time()
    print("solution time: ",(end - start))
    
    theta1, theta2 = sol[:,0], sol[:,2]
    theta1dot, theta2dot = sol[:,1], sol[:,3]

    # phase portrait of each generalized coordinate
    plt.subplot(2, 1, 1)
    plt.plot(theta1,theta1dot,'b')
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')

    plt.subplot(2, 1, 2)
    plt.plot(theta2,theta2dot,'r')
    plt.xlabel('Theta 2')
    plt.ylabel('Theta 2 Dot')
    plt.axis('equal')

    plt.show()

    # visualizer taken from:
    # https://scipython.com/blog/the-double-pendulum/
    def animateGIF():
        images = []
        # Convert to Cartesian coordinates of the two bob positions.
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)

        # Plotted bob circle radius
        r = 0.05
        # Plot a trail of the m2 bob's position for the last trail_secs seconds.
        trail_secs = 1
        # This corresponds to max_trail time points.
        max_trail = int(trail_secs / dt)

        def make_plot(i):
            # Plot and save an image of the double pendulum configuration for time
            # point i.
            # The pendulum rods.
            ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
            # Circles representing the anchor point of rod 1, and bobs 1 and 2.
            c0 = Circle((0, 0), r/2, fc='k', zorder=10)
            c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
            c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
            ax.add_patch(c0)
            ax.add_patch(c1)
            ax.add_patch(c2)

            # The trail will be divided into ns segments and plotted as a fading line.
            ns = 20
            s = max_trail // ns

            for j in range(ns):
                imin = i - (ns-j)*s
                if imin < 0:
                    continue
                imax = imin + s + 1
                # The fading looks better if we square the fractional length along the
                # trail.
                alpha = (j/ns)**2
                ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                        lw=2, alpha=alpha)

            # Centre the image on the fixed anchor point, and ensure the axes are equal
            ax.set_xlim(-l1-l2-r, l1+l2+r)
            ax.set_ylim(-l1-l2-r, l1+l2+r)
            ax.set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
            images.append(imageio.imread('frames/_img{:04d}.png'.format(i//di)))
            plt.cla()


        # Make an image every di time points, corresponding to a frame rate of fps
        # frames per second.
        # Frame rate, s-1
        fps = 10
        di = int(1/fps/dt)
        fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
        ax = fig.add_subplot(111)

        for i in range(0, tSpanExplicit.size, di):
            print(end="\r{} / {}".format(i // di, tSpanExplicit.size // di))
            make_plot(i)
        imageio.mimsave('gifs/doublePendulum.gif', images)

    animateGIF()