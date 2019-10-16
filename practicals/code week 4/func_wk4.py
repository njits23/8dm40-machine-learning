# -*- coding: utf-8 -*-
import numpy as np
import gryds

def brightness_aug(train_images, train_segmentations):
    mult=6;   #number of augmented images generated from each original image 
    sh=np.shape(train_images);
    images=np.zeros((sh[0]*mult,sh[1],sh[2],sh[3]))
    segmentations=np.zeros((sh[0]*mult,sh[1],sh[2],1))
    ba_range=np.linspace(0.8,1.2,mult)   #different multipliers for the images
    
    for idx, image in enumerate(train_images):
        segmentations[idx*mult:(idx+1)*mult]=train_segmentations[idx]
        for step,factor in enumerate(ba_range):
            images[idx*mult+step]=train_images[idx]*factor

    return images, segmentations

def geometric_aug(train_images, train_segmentations):
        
    mult=6;   #number of augmented images generated from each original image 
    sh=np.shape(train_images);
    images=np.zeros((sh[0]*mult,sh[1],sh[2],sh[3]))
    segmentations=np.zeros((sh[0]*mult,sh[1],sh[2],1))
    random_grids = np.zeros((mult,2,3,3))
    
    for trans_idx in range(mult):
            
        # Define a random 3x3 B-spline grid for a 2D image
        random_grid = np.random.rand(2, 3, 3)
        random_grid -= 0.5
        random_grid /= 20
        random_grids[trans_idx,:,:,:] = random_grid
        
    for idx, image in enumerate(train_images):
        for trans_idx in range(mult):
        
            # Define a B-spline transformation object
            bspline = gryds.BSplineTransformation(random_grids[trans_idx,:,:,:])
            
            interpolator_seg = gryds.Interpolator(train_segmentations[idx,:,:,0])
            segmentations[idx*mult+trans_idx,:,:,0] = interpolator_seg.transform(bspline)
            
            for channel in range(sh[3]):
                # Define an interpolator object for the image
                interpolator_image = gryds.Interpolator(train_images[idx,:,:,channel])
                    
                # Transform the image and add to the new images
                images[idx*mult+trans_idx,:,:,channel] = interpolator_image.transform(bspline)            
        
    return images, segmentations
