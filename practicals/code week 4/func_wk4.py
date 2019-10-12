import numpy as np

def brightness_aug(train_images, train_segmentations):
    mult=7;   #number of augmented images generated from each original image 
    sh=np.shape(train_images);
    images=np.zeros((sh[0]*mult,sh[1],sh[2],sh[3]))
    segmentations=np.zeros((sh[0]*mult,sh[1],sh[2],1))
    ba_range=np.linspace(0,1.2,mult)   #different multipliers for the images
    
    for idx, image in enumerate(train_images):
        segmentations[idx*mult:(idx+1)*mult]=train_segmentations[idx]
        for step,factor in enumerate(ba_range):
            images[idx*mult+step]=train_images[idx]*factor

    return images, segmentations
