#!/usr/bin/env python
import random
import math
import os
import argparse

import numpy as np

import PIL

from soccer_field import Field

def make_dataset(
    output_directory,
    num_points,
    seed=None,
    out_of_bounds=200,
    resolution=32
):

    if seed is not None:
        random.seed(seed)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
    beta = np.diag([np.deg2rad(5)**2])
    
    env = Field(alphas, beta, gui=False)
    
    x_min = env.MARKER_OFFSET_X - out_of_bounds
    x_max = env.MARKER_OFFSET_X + env.MARKER_DIST_X + out_of_bounds
    y_min = env.MARKER_OFFSET_Y - out_of_bounds
    y_max = env.MARKER_OFFSET_Y + env.MARKER_DIST_Y + out_of_bounds
    
    for i in range(num_points):
        x = (random.random() * (x_max - x_min) + x_min)
        y = (random.random() * (y_max - y_min) + y_min)
        theta = random.random() * math.pi * 2.
        observations = [env.observe([x,y,theta],j) for j in range(1,7)]
        q = env.p.getQuaternionFromEuler([0,0,theta])
        env.move_robot([x,y,theta])
        
        rgb_strip = env.render_panorama(resolution=resolution)
        
        image_name = '%s/rgb_%06i.png'%(output_directory, i)
        image = PIL.Image.fromarray(rgb_strip)
        image.save(image_name)
        print('Saved: %s'%image_name)
        
        label_name = '%s/label_%06i.npy'%(output_directory, i)
        label = np.array(observations)
        np.save(label_name, label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=32)
    
    args = parser.parse_args()
    if args.size is None:
        if args.split == 'train':
            args.size = 10000
        else:
            args.size = 1000
    
    # current train dataset uses seed 1234
    # current test dataset uses seed 12345
    
    make_dataset(
        'hw1_%s_dataset'%args.split,
        args.size,
        seed=args.seed,
        resolution=args.resolution,
    )
