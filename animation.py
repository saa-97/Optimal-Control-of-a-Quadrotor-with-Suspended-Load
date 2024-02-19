'''
    This file contains functions to animate the results of the simulation.
'''

import dynamics as dyn    
import numpy as np
from matplotlib import transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from scipy.ndimage import rotate
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec


def animate(xx_reg, xx_ref):

    dt = dyn.dt
    TT = dyn.TT
    tf = dyn.tf

    dt = dyn.dt
    tt_hor = np.linspace(0, tf, TT)
    time = np.arange(len(tt_hor)) * dt

    x_margin = 0.2
    y_margin = 0.3

    # Main plot for drone animation
    fig_main = plt.figure(figsize=(10, 6))
    ax_drone = fig_main.add_subplot(111, autoscale_on=False)
    ax_drone.grid(True)
    ax_drone.set_xlim(np.min(xx_reg[0]) - x_margin, np.max(xx_reg[0]) + x_margin)
    ax_drone.set_ylim(np.min(xx_reg[1]) - y_margin, np.max(xx_reg[1]) + y_margin)
    ax_drone.set_aspect('equal', adjustable='box') 
    ax_drone.set_yticklabels([])
    ax_drone.set_xticklabels([])
    ax_drone.title.set_text('Animation of the Trajectory')

    # Calculate the size of the drone rectangle based on the plot size
    drone_height = 0.05  # Adjust the drone height as needed
    drone_width = 0.5  # Adjust the drone width as needed
    drone_extent = (-drone_height/2, drone_height/2, -drone_width/2, drone_width/2)

    drone = ax_drone.add_patch(plt.Rectangle((0, 0), drone_width, drone_height, facecolor='blue', edgecolor='red', lw=3))

    # Add pendulum bob
    pendulum_length = 0.2  # Length of the pendulum string
    pendulum_pos = (xx_reg[0, 0], xx_reg[1, 0] - drone_height/2 - pendulum_length)
    pendulum = plt.Line2D([xx_reg[0, 0], pendulum_pos[0]], [xx_reg[1, 0] - drone_height/2, pendulum_pos[1]], color='black')

    bob_radius = 0.05  # The radius of the bob
    bob = plt.Circle((pendulum_pos[0], pendulum_pos[1]), bob_radius, color='black')  # Create the bob as a circle at the end of the pendulum
    
    ax_drone.add_patch(bob)  # Add the bob to the drone axes
    ax_drone.add_line(pendulum)

    time_template = 't = %.1f s'
    time_text = ax_drone.text(0.05, 0.9, '', transform=ax_drone.transAxes)
    fig_main.gca().set_aspect('equal', adjustable='box')

    fig_main.gca().grid(which='both')
    fig_main.gca().plot(xx_reg[0], xx_reg[1], c='b')
    fig_main.gca().plot(xx_ref[0], xx_ref[1], c='g', dashes=[2, 1])

    point1, = fig_main.gca().plot([], [], 'o', lw=2, c='b')

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Evolution of the positions and velocities', fontsize=16)  

    # Subplot for X Position
    ax2 = fig.add_subplot(gs[0, 0], aspect='auto')
    ax2.set_xlim(0, max(time))
    ax2.set_ylim(np.min(xx_reg[0]) - x_margin, np.max(xx_reg[0]) + x_margin)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X Position')
    line2, = ax2.plot([], [], lw=2)

    # Subplot for Y Position
    ax3 = fig.add_subplot(gs[0, 1], aspect='auto')
    ax3.set_xlim(0, max(time))
    ax3.set_ylim(np.min(xx_reg[1]) - y_margin, np.max(xx_reg[1]) + y_margin)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Y Position')
    line3, = ax3.plot([], [], lw=2)

    # Subplot for X Velocity
    ax4 = fig.add_subplot(gs[1, 0], aspect='auto')
    ax4.set_xlim(0, max(time))
    ax4.set_ylim(np.min(xx_reg[4]) - x_margin, np.max(xx_reg[4]) + x_margin)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('X Velocity')
    line4, = ax4.plot([], [], lw=2)

    # Subplot for Y Velocity
    ax5 = fig.add_subplot(gs[1, 1], aspect='auto')
    ax5.set_xlim(0, max(time))
    ax5.set_ylim(np.min(xx_reg[5]) - y_margin, np.max(xx_reg[5]) + y_margin)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Y Velocity')
    line5, = ax5.plot([], [], lw=2)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # Add x and y axis values
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax4.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax5.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax5.yaxis.set_major_locator(MaxNLocator(nbins=5))

    plt.tight_layout()

    def init():
        drone.set_xy((xx_reg[0, 0] - drone_width/2, xx_reg[1, 0] - drone_height/2))
        drone.set_angle(np.degrees(xx_reg[3, 0]))

        point1.set_data(xx_reg[0, 0], xx_reg[1, 0])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])

        time_text.set_text('')

        pendulum.set_data([], [])
        bob.center = (pendulum_pos[0], pendulum_pos[1])

        return time_text, point1, drone, pendulum, bob
    
    def animate(i):
        drone.set_xy((xx_reg[0, i] - drone_width/2, xx_reg[1, i] - drone_height/2))
        drone.set_angle(np.degrees(xx_reg[3, i]))

        point1.set_data(xx_reg[0, i], xx_reg[1, i])
        thisx = np.append(line2.get_xdata(), time[i])
        thisy = np.append(line2.get_ydata(), xx_reg[0, i])
        line2.set_data(thisx, thisy)
        thisx = np.append(line3.get_xdata(), time[i])
        thisy = np.append(line3.get_ydata(), xx_reg[1, i])
        line3.set_data(thisx, thisy)
        thisx = np.append(line4.get_xdata(), time[i])
        thisy = np.append(line4.get_ydata(), xx_reg[4, i])
        line4.set_data(thisx, thisy)
        thisx = np.append(line5.get_xdata(), time[i])
        thisy = np.append(line5.get_ydata(), xx_reg[5, i])
        line5.set_data(thisx, thisy)

        time_text.set_text(time_template % (i * dt))

        # Update pendulum position and angle
        pendulum_angle = np.degrees(xx_reg[2, i])
        pendulum_length = 0.2  # Length of the pendulum string
        pendulum_pos = (xx_reg[0, i] - np.sin(np.radians(pendulum_angle)) * pendulum_length,
                        xx_reg[1, i] - drone_height/2 - np.cos(np.radians(pendulum_angle)) * pendulum_length)

        pendulum.set_data([xx_reg[0, i], pendulum_pos[0]], [xx_reg[1, i] - drone_height/2, pendulum_pos[1]])
        bob.center = pendulum_pos

        return time_text, point1, line2, line3, line4, line5, drone, pendulum, bob
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    animation_obj = animation.FuncAnimation(fig_main, animate, init_func=init, frames=range(len(xx_reg[0])), interval=0.001, blit=True)
    plt.show()
    return animation_obj
