#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:57:04 2024

@author: senyuz
"""

import logging
import math
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import skimage
import warnings

log = logging.getLogger(__name__)

# Several skimage calls (e.g. remove_small_objects' min_size) emit FutureWarnings
# about parameter renames. The calls are correct for the pinned version, so we
# silence that category to keep the workflow output readable.
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.optimize import linear_sum_assignment
from simplification.cutil import simplify_coords_vw
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize


def segmentation_skeleton(image, sigma, small,thresh_top):
    '''
    Function for preprocessing and skeletonization of image-data.
    
    Parameters
    ----------
    image : array
        Image input.
    sigma : float
        Amount to perform gaussina spread with.
    small : float
        Removal of small coherent pixel groups.
    thresh_top : float
        lower bound of threshold  value.

    Returns
    -------
    ImF, imageCleaned, imH.

    '''
    # 1) gaussian filter
    imG=skimage.filters.gaussian(image,sigma)
    # 2) frangi tubeness
    imF = skimage.filters.frangi(imG, sigmas=np.arange(1, 2, 0.1), scale_step=0.1, alpha=0.1, beta=2, gamma=15, black_ridges=False, mode='reflect', cval=0)
    # 3) morph grey closing with SE of disk on r=2
    #circle = skimage.morphology.selem.disk(2)
    #imC = skimage.morphology.grey.closing(imF, circle)
    # 4) CLAHE
    imE = skimage.exposure.equalize_adapthist(imF, kernel_size=None, clip_limit=0.01, nbins=256)
    # 5) median filter
    imM = sp.ndimage.median_filter(imE, size=(2,2),  mode='reflect', cval=0.0, origin=0)
    # 6) hysteresis
    thresh = threshold_otsu(imM)
    imH = skimage.filters.apply_hysteresis_threshold(imE, thresh*thresh_top, thresh)
    # 7) morph grey closing again
    #imC2 =skimage.morphology.grey.closing(imH*1, circle)
    # 8) skeletonize
    imS = skimage.morphology.skeletonize(imH > 0)
    # 9) remove small objects
    imageCleaned = skimage.morphology.remove_small_objects(imS, small, connectivity=2) > 0

    ##
    imageHysteresisCleaned = skimage.morphology.remove_small_objects(imH, small, connectivity=2) > 0
    ##
    return(imF,imageCleaned,imH,imageHysteresisCleaned)

def segmentation_skeleton_short(image, sigma, small, thresh_top):
    """
    Simplified segmentation and skeletonization function.
    
    This function performs the following steps:
      1. Binarizes the normalized input image (using image > 0).
      2. Skeletonizes the binarized image.
      3. Removes small objects from the skeletonized image.
      4. Removes small objects from the binarized image.
      
    The parameters `sigma` and `thresh_top` are retained for compatibility but are not used.
    
    Parameters
    ----------
    image : ndarray
        Input image (assumed to be normalized).
    sigma : float
        Dummy parameter (not used in this simplified version).
    small : float
        Minimum size (in pixels) for connected components to keep.
    thresh_top : float
        Dummy parameter (not used in this simplified version).
    
    Returns
    -------
    imF : ndarray
        Binarized image (serving as the tubeness output).
    imageCleaned : ndarray
        Skeletonized image after removal of small objects.
    imH : ndarray
        Binarized image (duplicate of imF for compatibility).
    imageHysteresisCleaned : ndarray
        Binarized image after removal of small objects.
    """
    # 1) Binarize the normalized input image.
    #    (Assumes that image is already normalized; thresholding with > 0.)
    imH = image > 0

    # Use the binarized image as the tubeness output.
    imF = imH.copy()

    # 2) Skeletonize the binary image.
    imageSkeletonize = skimage.morphology.skeletonize(imH)

    # 3) Remove small objects from the skeletonized image.
    imageCleaned = skimage.morphology.remove_small_objects(
        imageSkeletonize, min_size=small, connectivity=2
    ) > 0

    # 4) Remove small objects from the binary image.
    imageHysteresisCleaned = skimage.morphology.remove_small_objects(
        imH, min_size=small, connectivity=2
    ) > 0

    return imF, imageCleaned, imH, imageHysteresisCleaned


def node_find(imageSkeleton):
    '''
    Locate and mark position of nodes for the skeletonized image

    Parameters
    ----------
    imageSkeleton : array
        Original skeletonized image.

    Returns
    -------
    imageNodes : array
        Skeletonized image with nodes.

    '''

    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(imageSkeleton)
    imageNodes=np.zeros(np.shape(imageSkeleton))

    M,N = imageSkeleton.shape
    # For each non-zero pixel...
    for (r,c) in zip(rows,cols):
        imageSkeleton[r-1:r+2,c-1:c+2]
        # Extract an 8-connected neighbourhood
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')
        if ((r+1)<M and (c+1)<N):
            imageSection = imageSkeleton[row_neigh,col_neigh]
        else:
            imageSection = np.array([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]])
        #remove the center value to label adjacent values to it
        imageSection[1,1] = 0
        imageLabeled, labels = sp.ndimage.label(imageSection)

        if ((labels != 0 and labels != 2) or ( np.sum(imageSection*1)>=4)):
            imageNodes[r,c]=1
    return imageNodes


def project_edges(imE,eps,size):
    '''
    Function to add additional nodes to graph using VW algorithm

    Parameters
    ----------
    imE : array
        Skeletonized image with nodes marked.
    eps : float
        VW threshold.
    size : float
        limit value.

    Returns
    -------
    filtF2 : array
        Skeletonized image containing additional nodes.
    blank2 : array
        all addintioanl nodes marked on blanck image.

    '''
    imEc = imE.copy()
    #add padding
    imEc = np.pad(imEc, 1, 'constant')
    filt = imEc.copy()
    filtF = imEc.copy()

    (rows,cols) = np.nonzero((imEc>1)*1)

    M,N = filt.shape
    filtU = imEc.copy()
    blank = np.zeros((M,N))

    edgeval = 3

    for k in range(len(rows)):

        line_coords = []
        counter = 0
        stop2 = 0
        stop3 = 0
        stop4 = 0
        r = rows[k]
        c = cols[k]

        line_coords.append((r,c))

        filt = imEc.copy()
        filtU[r,c] = 0

        filt[r,c] = edgeval
        #filt[pos[edgesL[k][1]][1],pos[edgesL[k][1]][0]] = edgeval

        # start test on intial node entry to see if multiple tracings has to be done.
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

        col_neighOri = col_neigh.astype('int')
        row_neighOri = row_neigh.astype('int')

        imageSection = filtU[row_neigh,col_neigh]

        imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)

        # need to have a check for if no value is over 1
        if((np.max(filtU)==1) and (labels4>0)):
            line_coords4 = line_coords.copy()
            # if this has happened, then there is a loop like structure left, so this tracing is different from rest
            (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')

            imageSection = filtU[row_neigh,col_neigh]

            imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)

            filtU4 = filtU.copy()
            filt4 = filt.copy()
            stop4 = 0
            filtU4[row_neigh,col_neigh] = (filtU4[row_neigh,col_neigh]==1)*1
            #imageLabeled4 = (imageLabeled4==(i+1))*1
            ind = np.where(imageLabeled4)
            index = 0
            if(len(ind[0])!=1):
                index = int(np.floor(len(ind)/2.))
            #move one
            r4 = (r+ind[0])[index]-1
            c4 = (c+ind[1])[index]-1
            line_coords4.append((r4,c4))

            filt4[r4,c4] = edgeval
            filtU4[r4,c4] = 0

            #check how it now looks
            (col_neigh,row_neigh) = np.meshgrid(np.array([c4-1,c4,c4+1]), np.array([r4-1,r4,r4+1]))

            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')

            imageSection = filtU4[row_neigh,col_neigh]
            imageLabeled4, labels4 = sp.ndimage.label(imageSection)

            while((labels4==1) and (stop4==0)):
                ind = np.where(imageLabeled4)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r4 = (r4+ind[0])[index]-1
                c4 = (c4+ind[1])[index]-1
                line_coords4.append((r4,c4))

                filt4[r4,c4] = edgeval
                filtU4[r4,c4] = 0

                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c4-1,c4,c4+1]), np.array([r4-1,r4,r4+1]))

                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')

                imageSection = filtU4[row_neigh,col_neigh]
                imageLabeled4, labels4 = sp.ndimage.label(imageSection)

                if(np.max(imageSection)==0):
                    counter += 1
                    stop4 = 1
                    filtU[rows[k],cols[k]]=1
                    filtU = filtU - (filt4 == edgeval)*1
                    # do the rdp algorithm
                    #ind_stack = np.column_stack(np.where(edgeImg==edgeval))
                    rdpInd = simplify_coords_vw(line_coords4, eps)
                    #there is only one, so need to ad one in
                    for i in range(len(rdpInd)-1):
                        if(imEc[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])]==1 ):
                            filtF[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
                            blank[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2




        if(len(np.nonzero(imageSection)[0])>0):
            if(np.sum(imageSection)/len(np.nonzero(imageSection)[0])==2):
                #this means that we are only neighbours to other nodes
                stop4=1

        # i will run the amounts of runs as there are labels
        if(stop4==0):
            line_coords3 = line_coords.copy()
            for i in range(labels4):

                filtU3 = filtU.copy()
                filt3 = filt.copy()
                stop3 = 0
                filtU3[row_neighOri,col_neighOri] = filtU3[row_neighOri,col_neighOri]*(imageLabeled4==(i+1))*1
                imageLabeled3 = (imageLabeled4==(i+1))*1
                ind = np.where(imageLabeled3)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r3 = (r+ind[0])[index]-1
                c3 = (c+ind[1])[index]-1
                line_coords3.append((r3,c3))

                filt3[r3,c3] = edgeval
                filtU3[r3,c3] = 0

                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))

                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')

                imageSection = filtU3[row_neigh,col_neigh]
                imageLabeled3, labels3 = sp.ndimage.label(imageSection)


                while((labels3==1) and (stop3==0)):
                    ind = np.where(imageLabeled3)
                    index = 0
                    if(len(ind[0])!=1):
                        index = int(np.floor(len(ind)/2.))
                    #move one
                    r3 = (r3+ind[0])[index]-1
                    c3 = (c3+ind[1])[index]-1
                    line_coords3.append((r3,c3))

                    filt3[r3,c3] = edgeval
                    filtU3[r3,c3] = 0

                    #check how it now looks
                    (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))

                    col_neigh = col_neigh.astype('int')
                    row_neigh = row_neigh.astype('int')

                    imageSection = filtU3[row_neigh,col_neigh]
                    imageLabeled3, labels3 = sp.ndimage.label(imageSection)
                    # NEW
                    if(np.max(imageSection)==0):
                        #check if a node exist in imEc
                        if(np.max(imEc[row_neigh,col_neigh])==2):
                            #the loop ends here
                            # this is the stopper of this while loop for one trace only
                            stop3 = 1
                            stop2 = 1
                            counter += 1

                            #filtU[rows[k],cols[k]]=1
                            filtU = filtU - (filt3 == edgeval)*1

                            # do the VW algorithm
                            #ind_stack = np.column_stack(np.where(edgeImg==edgeval))
                            rdpInd = simplify_coords_vw(line_coords3, eps)
                            #if(len(rdpInd)>2):
                            for i in range(len(rdpInd)-1):
                                if(imEc[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])]==1 ):
                                    filtF[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
                                    blank[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2

                    elif((np.max(imageSection)>1)):
                        ind = np.where(imageSection==2)
                        index = 0
                        #move one
                        r3 = (r3+ind[0])[index]-1
                        c3 = (c3+ind[1])[index]-1
                        line_coords3.append((r3,c3))

                        filt3[r3,c3] = edgeval
                        filtU3[r3,c3] = 0

                        # this is the stopper of this while loop for one trace only
                        counter += 1
                        stop3 = 1
                        stop2 = 1
                        filtU[rows[k],cols[k]]=1
                        filtU = filtU - (filt3 == edgeval)*1
                        # do the VW algorithm
                        #ind_stack = np.column_stack(np.where(edgeImg==edgeval))
                        rdpInd = simplify_coords_vw(line_coords3, eps)
                        #if(len(rdpInd)>2):
                        for i in range(len(rdpInd)-1):
                            if(imEc[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])]==1 ):
                                filtF[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
                                blank[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2


    #* Extension: Now handle connected components without pre-marked nodes
    structure = np.ones((3, 3))
    labeled, nr_objects = sp.ndimage.label(filtU == 1, structure=structure)
    for obj_label in range(1, nr_objects + 1):
        component = (labeled == obj_label)
        if np.sum(component) > size:
            # Start from an arbitrary point in the component
            indl = np.where(component)
            r_start, c_start = indl[0][0], indl[1][0]
            line_coords = []
            visited = np.zeros_like(component, dtype=bool)
            stack = [(r_start, c_start)]
            while stack:
                r_curr, c_curr = stack.pop()
                if not visited[r_curr, c_curr]:
                    visited[r_curr, c_curr] = True
                    line_coords.append((r_curr, c_curr))
                    # Get neighbors by looking at 8-connected neighborhood
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            # Skip the current/center pixel
                            if dr == 0 and dc == 0:
                                continue
                            # Update to the neighbor pixel coordinates
                            r_next = r_curr + dr
                            c_next = c_curr + dc
                            # Check if the neighbor is within the component and not visited
                            if (0 <= r_next < component.shape[0] and
                                0 <= c_next < component.shape[1] and
                                component[r_next, c_next] and not visited[r_next, c_next]):
                                stack.append((r_next, c_next))
            # Now, line_coords contains all coordinates in the component
            # Apply the VW algorithm
            rdpInd = simplify_coords_vw(line_coords, eps)
            # Update filtF and blank
            for coord in rdpInd:
                r_coord, c_coord = coord
                if imEc[int(r_coord), int(c_coord)] == 1:
                    filtF[int(r_coord), int(c_coord)] = 2
                    blank[int(r_coord), int(c_coord)] = 2
            # Remove the processed component from filtU
            filtU[component] = 0
    ##

    #remove padding
    filtF2 = filtF[1:-1, 1:-1]
    blank2 = blank[1:-1, 1:-1]

    return filtF2,blank2

#-------------------------------------------------------------------------------
# Add additional nodes where thickness and intensity change drastically.
# Take into account the minimum length of pixel change before change and merge close nodes.

def get_ordered_chain(cc_mask, cc_pixels):
    """
    Given a mask `cc_mask` for a connected component (True where pixels belong),
    and a list of those pixel coordinates `cc_pixels`, produce an ordered chain
    of pixels from an endpoint to another endpoint.

    This assumes the connected component is basically a single line/curve
    or at most a single branching that we ignore. If there's complex branching,
    you'd need a more complex approach.
    """

    if len(cc_pixels) == 0:
        return []

    # Convert list of pixels to a set for quick membership tests
    cc_set = set(tuple(p) for p in cc_pixels)

    # 1) Identify endpoints: those pixels with exactly 1 neighbor in cc_mask
    endpoints = []
    for (r, c) in cc_pixels:
        neighbor_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if (rr, cc) in cc_set:
                    neighbor_count += 1
        # Endpoint if it has only 1 neighbor in cc_mask
        if neighbor_count == 1:
            endpoints.append((r, c))

    # If no endpoints (e.g. it might be a loop), just pick the first pixel
    if len(endpoints) == 0:
        # fallback: pick the first pixel in cc_pixels as a start
        endpoints.append(tuple(cc_pixels[0]))

    # 2) Use BFS/DFS from the first endpoint to build an ordered chain
    start = endpoints[0]
    visited = set()
    ordered_chain = []
    current = start

    while True:
        ordered_chain.append(current)
        visited.add(current)
        # Find the next neighbor
        neighbors = []
        (r0, c0) = current

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                rr = r0 + dr
                cc = c0 + dc
                if (rr, cc) in cc_set and (rr, cc) not in visited:
                    neighbors.append((rr, cc))

        if len(neighbors) == 0:
            # No unvisited neighbors => we reached the end
            break
        elif len(neighbors) == 1:
            # Exactly one way forward => follow it
            current = neighbors[0]
        else:
            # More than one => branching
            # For a simple chain, pick the first or handle branching separately
            current = neighbors[0]

    return ordered_chain


def maybe_merge_with_existing(r, c, skel, radius):
    """
    If an existing node is within 'radius' pixels of (r,c), return that node’s coords.
    Else return (r,c).
    """
    (h, w) = skel.shape
    # define bounding region
    rmin = max(0, r-radius)
    rmax = min(h, r+radius+1)
    cmin = max(0, c-radius)
    cmax = min(w, c+radius+1)

    # search for existing node within that region
    sub = skel[rmin:rmax, cmin:cmax]
    coords = np.argwhere(sub>1)  # any pixel labeled 2 or more is a "node"
    if len(coords)==0:
        return (r,c)  # no existing node found

    # pick the closest node
    best_dist = 9999
    best_point = (r,c)
    for (rr,cc) in coords:
        # coords are local to sub => shift
        R = rmin + rr
        C = cmin + cc
        d = (R - r)**2 + (C - c)**2
        if d < best_dist:
            best_dist = d
            best_point = (R, C)
    return best_point


def insert_nodes_by_thickness_intensity_dynamic(
    skeleton_image,
    thickness_map,
    intensity_map,
    min_edge_pixels=10,
    merge_radius=3,
    thickness_factor=1.5,
    intensity_factor=1.5,
    thickness_thresh=2.0,
    intensity_thresh=0.25
):
    """
    Post-processing step:
      - For each connected component, compute local thickness & intensity stats (e.g. mean+std).
      - Use those stats to define thresholds for that component.
      - Insert a node if local difference exceeds that threshold.

    :param thickness_factor: multiplier for thickness standard deviation (or MAD).
    :param intensity_factor: multiplier for intensity standard deviation (or MAD).
    """

    new_skel = skeleton_image.copy()
    labeled_cc, n_cc = sp.ndimage.label((skeleton_image > 0).astype(np.uint8))

    for cc_idx in range(1, n_cc+1):
        cc_mask = (labeled_cc == cc_idx)

        # All pixels in this connected component
        cc_pixels = np.argwhere(cc_mask)
        if len(cc_pixels) < 2:
            continue

        # --- 1) Gather thickness/intensity stats for this CC ---
        thickness_vals = []
        intensity_vals = []
        for (r, c) in cc_pixels:
            thickness_vals.append(thickness_map[r, c])
            intensity_vals.append(intensity_map[r, c])

        thickness_vals = np.array(thickness_vals)
        intensity_vals = np.array(intensity_vals)

        # Compute local statistics (example: mean + std)
        mean_th = np.mean(thickness_vals)
        std_th  = np.std(thickness_vals)
        mean_i  = np.mean(intensity_vals)
        std_i   = np.std(intensity_vals)

        # Avoid zeros
        if std_th < 1e-9:
            std_th = 1e-9
        if std_i < 1e-9:
            std_i = 1e-9

        # Define thresholds for thickness and intensity differences
        # Example: "difference > thickness_factor * std_th"
        # Or you could do ratio-based if that’s more appropriate.
        thickness_thresh = thickness_factor * std_th
        intensity_thresh = intensity_factor * std_i

        # --- 2) BFS/DFS to get the "chain" (same logic as before) ---
        # ... create an ordered path from an endpoint ...
        # (Use the BFS/DFS logic from your existing code or from the previous example)

        ordered_pixels = get_ordered_chain(cc_mask, cc_pixels)  # your BFS/DFS

        # --- 3) Walk along the chain, compare differences ---
        last_break_idx = 0
        for i in range(len(ordered_pixels)-1):
            (r1,c1) = ordered_pixels[i]
            (r2,c2) = ordered_pixels[i+1]

            if (i - last_break_idx) < min_edge_pixels:
                continue  # skip if too few pixels from last break

            thick1 = thickness_map[r1,c1]
            thick2 = thickness_map[r2,c2]
            delta_t = abs(thick2 - thick1)

            i1 = intensity_map[r1,c1]
            i2 = intensity_map[r2,c2]
            # Just do absolute difference vs local threshold
            delta_i = abs(i2 - i1)

            if (delta_t >= thickness_thresh) or (delta_i >= intensity_thresh):
                # Insert new node
                (nr, nc) = maybe_merge_with_existing(r2, c2, new_skel, merge_radius)
                new_skel[nr,nc] = 2  # mark as node
                last_break_idx = i+1

    return new_skel







#-------------------------------------------------------------------------------


def project_mask(imF):
    '''
    Create mask containing each edge marked with a unique value
    
    Parameters
    ----------
    imF : array
        Skeletonized image with marked nodes.

    Returns
    -------
    mask2 : array
        DESCRIPTION.
    index_list : list
        list of pixels for mask.

    '''
    # add padding
    img = np.pad(imF, 1, 'constant')

    filt = img.copy()

    mask = np.zeros((img.shape))

    (rows,cols) = np.nonzero((img>1)*1)

    M,N = filt.shape

    edgeval = 3
    count_mask = 4
    index_list = []

    for k in range(len(rows)):

        line_coords = []
        counter = 0
        stop3 = 0
        stop4 = 0
        r = rows[k]
        c = cols[k]

        line_coords.append((r,c))

        filt = img.copy()
        filtU = img.copy()
        filtU[r,c] = 0
        filt[r,c] = edgeval

        # start test on intial node entry to see if multiple tracings has to be done.
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

        col_neighOri = col_neigh.astype('int')
        row_neighOri = row_neigh.astype('int')

        imageSection = filtU[row_neighOri,col_neighOri]

        imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)

        # need to have a check for if no value is over 1
        if((np.max(filtU)==1) and (labels4>0)):
            line_coords4 = line_coords.copy()
            # if this has happened, then there is a loop like structure left, so this tracing is different from rest
            (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')

            imageSection = filtU[row_neigh,col_neigh]

            imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)

            filtU4 = filtU.copy()
            filt4 = filt.copy()
            stop4 = 0
            filtU4[row_neigh,col_neigh] = (filtU4[row_neigh,col_neigh]==1)*1
            #imageLabeled4 = (imageLabeled4==(i+1))*1
            ind = np.where(imageLabeled4)
            index = 0
            if(len(ind[0])!=1):
                index = int(np.floor(len(ind)/2.))
            #move one
            r4 = (r+ind[0])[index]-1
            c4 = (c+ind[1])[index]-1
            line_coords4.append((r4,c4))

            filt4[r4,c4] = edgeval
            filtU4[r4,c4] = 0

            #check how it now looks
            (col_neigh,row_neigh) = np.meshgrid(np.array([c4-1,c4,c4+1]), np.array([r4-1,r4,r4+1]))

            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')

            imageSection = filtU4[row_neigh,col_neigh]
            imageLabeled4, labels4 = sp.ndimage.label(imageSection)

            while((labels4==1) and (stop4==0)):
                ind = np.where(imageLabeled4)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r4 = (r4+ind[0])[index]-1
                c4 = (c4+ind[1])[index]-1
                line_coords4.append((r4,c4))

                filt4[r4,c4] = edgeval
                filtU4[r4,c4] = 0

                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c4-1,c4,c4+1]), np.array([r4-1,r4,r4+1]))

                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')

                imageSection = filtU4[row_neigh,col_neigh]
                imageLabeled4, labels4 = sp.ndimage.label(imageSection)

                if(np.max(imageSection)==0):
                    counter += 1
                    stop4 = 1
                    filtU[rows[k],cols[k]]=1
                    filtU = filtU - (filt4 == edgeval)*1

                    # mark out traced line in mask
                    node1 = line_coords4[0]
                    node2 = r4,c4
                    mask[node2] = 2
                    count_mask += 1
                    ks = 0
                    index_list.append((np.subtract(node1,1),np.subtract(node2,1),count_mask))
                    while(node2!=line_coords4[ks]):
                        mask[line_coords4[ks]] = count_mask
                        ks += 1



        if(len(np.nonzero(imageSection)[0])>0):
            if(np.sum(imageSection)/len(np.nonzero(imageSection)[0])==2):
                #this means that we are only neighbours to other nodes
                stop4=1

        # i will run the amounts of runs as there are labels
        if(stop4==0):

            for i in range(labels4):
                line_coords3 = line_coords.copy()
                filtU3 = filtU.copy()
                filt3 = filt.copy()
                stop3 = 0
                filtU3[row_neighOri,col_neighOri] = filtU3[row_neighOri,col_neighOri]*(imageLabeled4==(i+1))*1
                imageLabeled3 = (imageLabeled4==(i+1))*1
                ind = np.where(imageLabeled3)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r3 = (r+ind[0])[index]-1
                c3 = (c+ind[1])[index]-1
                line_coords3.append((r3,c3))

                filt3[r3,c3] = edgeval
                filtU3[r3,c3] = 0

                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))

                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')

                imageSection = filtU3[row_neigh,col_neigh]
                imageLabeled3, labels3 = sp.ndimage.label(imageSection)

                fS=1

                while(((labels3==1) and (stop3==0)) or ((fS==1) and (labels3>0))):
                    fS=0
                    ind = np.where(imageLabeled3)
                    index = 0
                    if(len(ind[0])!=1):
                        index = int(np.floor(len(ind)/2.))
                    #move one
                    r3 = (r3+ind[0])[index]-1
                    c3 = (c3+ind[1])[index]-1
                    line_coords3.append((r3,c3))

                    filt3[r3,c3] = edgeval
                    filtU3[r3,c3] = 0

                    #check how it now looks
                    (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))

                    col_neigh = col_neigh.astype('int')
                    row_neigh = row_neigh.astype('int')

                    imageSection = filtU3[row_neigh,col_neigh]
                    imageLabeled3, labels3 = sp.ndimage.label(imageSection)
                    if(np.max(imageSection)==0):
                        #check if a node exist in imEc
                        if(np.max(img[row_neigh,col_neigh])==2):
                            #the loop ends here
                            # this is the stopper of this while loop for one trace only
                            counter += 1
                            stop3 = 1
                            node1 = line_coords3[0]
                            node2 = r3,c3
                            mask[node2] = 2
                            count_mask += 1
                            ks = 0
                            index_list.append((np.subtract(node1,1),np.subtract(node2,1),count_mask))
                            while(node2!=line_coords3[ks]):
                                mask[line_coords3[ks]] = count_mask
                                ks += 1

                    elif(np.max(imageSection)>1):
                        ind = np.where(imageSection==2)
                        index = 0
                        #move one
                        r3 = (r3+ind[0])[index]-1
                        c3 = (c3+ind[1])[index]-1
                        line_coords3.append((r3,c3))

                        filt3[r3,c3] = edgeval
                        filtU3[r3,c3] = 0

                        # this is the stopper of this while loop for one trace only
                        counter += 1
                        stop3 = 1
                        # mark out traced line in mask
                        node1 = line_coords3[0]
                        node2 = r3,c3
                        mask[node2] = 2
                        count_mask += 1
                        ks = 0
                        index_list.append((np.subtract(node1,1),np.subtract(node2,1),count_mask))
                        while(node2!=line_coords3[ks]):
                            mask[line_coords3[ks]] = count_mask
                            ks += 1


    #remove padding
    mask2 = mask[1:-1, 1:-1]
    return mask2,index_list

def node_condense(imageFiltered,imageSkeleton,kernel):
    '''
    Condensation of nodes based on kernel size set by user.

    Parameters
    ----------
    imageFiltered : array
        Skeletonized image containing nodes and the added nodes from VW algorithm.
    imageSkeleton : array
        Original skeletonized image with nodes marked.
    kernel : array
        kernel to filter on.

    Returns
    -------
    Skeletonized image with nodes defined and merged together based on distance.

    '''

    imageLabeled, labels = sp.ndimage.label(imageFiltered, structure=np.ones((3, 3)))
    #need to have rolling window to condense nodes together
    imgSL = imageLabeled+imageSkeleton

    half = int(len(kernel)/2)
    M,N = imgSL.shape
    for l in range(half,M-half):
        for k in range(half,N-half):

            small = imgSL[l-half:l+half,k-half:k+half]
            # get all pixel location for vals higher than 1
            if((np.sum((small>1)*1)>2)):
                location=np.argwhere(small > 1)
                #if any endpoints,remove those.
                for z in range(len(location)):
                    coord1 = np.array([location[z,0],location[z,1]])
                    if(np.sum((imgSL[l-half+coord1[0]-1:l-half+coord1[0]+2,k-half+coord1[1]-1:k-half+coord1[1]+2]>0)*1)==2):
                        #this is an endpoint, remove it
                        imgSL[l-half+coord1[0],k-half+coord1[1]] = 1
                    small = imgSL[l-half:l+half,k-half:k+half]
                location=np.argwhere(small > 1)
                for i in range(len(location)):
                    if(i != (np.floor(len(location)/2))):
                        #this is the middle one , need to keep it
                        coord1 = np.array([location[i,0],location[i,1]])
                        imgSL[l-half+coord1[0],k-half+coord1[1]] = 1

            # there are two different points close to each other
            elif(np.sum((small>1)*1)==2):

                location=np.argwhere(small > 1)

                # test that both points are not endnodes
                coord1 = np.array([location[0,0],location[0,1]])
                coord2 = np.array([location[1,0],location[1,1]])

                node1conn = np.sum((imgSL[l-half+coord1[0]-1:l-half+coord1[0]+2,k-half+coord1[1]-1:k-half+coord1[1]+2]>0)*1)
                node2conn = np.sum((imgSL[l-half+coord2[0]-1:l-half+coord2[0]+2,k-half+coord2[1]-1:k-half+coord2[1]+2]>0)*1)
                if((node1conn>2) & (node2conn>2)):
                    # nod end nodes
                    y_min = min(l-half+location[0][0],l-half+location[1][0])
                    y_max = max(l-half+location[0][0],l-half+location[1][0])
                    x_min = min(k-half+location[0][1], k-half+location[1][1])
                    x_max = max(k-half+location[0][1], k-half+location[1][1])
                    zoom = (imgSL[y_min:y_max+1,x_min:x_max+1]>0)*1
                    conV = skimage.measure.label(zoom, connectivity=2)
                    if(np.max(conV)==1):
                        com = ndimage.center_of_mass(conV)

                        if(coord1[1]<coord2[1]):
                            coordinateC = math.ceil(com[0])+coord1[0],math.ceil(com[1])+coord1[1]
                            coordinateF = math.floor(com[0])+coord1[0],math.floor(com[1])+coord1[1]
                        else:
                            coordinateC = math.ceil(com[0])+coord1[0],-math.ceil(com[1])+coord1[1]
                            coordinateF = math.floor(com[0])+coord1[0],-math.floor(com[1])+coord1[1]

                        # test if the coordinate has value different from 0
                        if((imgSL[l-half+coordinateC[0],k-half+coordinateC[1]])>0):
                            #contnue set value to one of the other values
                            val = imgSL[l-half+coord1[0],k-half+coord1[1]]
                            imgSL[l-half+location[0][0],k-half+location[0][1]] = 1
                            imgSL[l-half+location[1][0],k-half+location[1][1]] = 1
                            imgSL[l-half+coordinateC[0],k-half+coordinateC[1]] = val

                        elif((imgSL[l-half+coordinateF[0],k-half+coordinateF[1]])>0):
                            val = imgSL[l-half+coord1[0],k-half+coord1[1]]
                            imgSL[l-half+location[0][0],k-half+location[0][1]] = 1
                            imgSL[l-half+location[1][0],k-half+location[1][1]] = 1
                            imgSL[l-half+coordinateF[0],k-half+coordinateF[1]] = val
                    elif(np.max(conV)==2):
                         # need to check that the two nodes are connected
                        arr,num = skimage.measure.label((small>0)*1,return_num=True, connectivity=2)
                        #if num = 2, then they are not together- meaning that its two seperated branches. if num=1 then together
                        if(num==1):
                            #remove end node
                            imgSL[l-half+coord2[0],k-half+coord2[1]] = 1
                else:
                    # need to check that the two nodes are connected
                    arr,num = skimage.measure.label((small>0)*1,return_num=True, connectivity=2)
                    #if num = 2, then they are not together- meaning that its two seperated branches. if num=1 then together
                    if(num==1):
                    #remove end node
                        if(node1conn==2):
                            imgSL[l-half+coord1[0],k-half+coord1[1]] = 1
                        if(node2conn==2):
                            imgSL[l-half+coord2[0],k-half+coord2[1]] = 1
    return (imgSL-imageSkeleton)


def condense_mask(index_list,imageNodeCondense,mask,size):
    '''
    Create list containing node positions

    Parameters
    ----------
    index_list : TYPE
        DESCRIPTION.
    imageNodeCondense : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    df_new : TYPE
        DESCRIPTION.

    '''
    if(index_list==[]):
        df_new = []

    else:
        len_index = np.zeros(len(index_list))
        correct_pos1 = [0]*len(index_list)
        correct_pos2 = [0]*len(index_list)
        nodenum1 = np.zeros(len(index_list))
        nodenum2 = np.zeros(len(index_list))
        posx,posy = np.nonzero((imageNodeCondense>0)*1)
        index1 = np.zeros(len(posx))
        index2 = np.zeros(len(posx))
        for i in range(len(index_list)):
            #if smaller than 10 in length, then remove this one
            len_index[i] = np.sqrt((index_list[i][0][0] - index_list[i][1][0] )**2 + (index_list[i][0][1] - index_list[i][1][1] )**2 )
            for j in range(len(posx)):
                index1[j] = np.sqrt((index_list[i][0][0] - posx[j] )**2 + (index_list[i][0][1] - posy[j] )**2 )
                index2[j] = np.sqrt((index_list[i][1][0] - posx[j] )**2 + (index_list[i][1][1] - posy[j] )**2 )
            argm1 = np.argmin(index1)
            argm2 = np.argmin(index2)
            correct_pos1[i] = list((posx[argm1],posy[argm1]))
            nodenum1[i] = argm1
            correct_pos2[i] = list((posx[argm2],posy[argm2]))
            nodenum2[i] = argm2

        index_pd = pd.DataFrame(index_list)
        index_pd[3] = len_index
        index_pd[4] = correct_pos1
        index_pd[5] = nodenum1
        index_pd[6] = correct_pos2
        index_pd[7] = nodenum2
        #df = index_pd[index_pd[3] > size]
        df= index_pd

        df = df[df[2].isin(np.unique(mask))]
        df = df.reset_index(drop=True)
        vals = np.unique(df[2])
        delete_row= []
        ss = df.duplicated(subset=[5,7], keep=False)
        index_ss = ss[ss].index
        delete_df = []
        for i in index_ss:
            smallM = []
            indexM = []
            df_sub = df.loc[(df[5] == df[5].iloc[i]) & (df[7]==df[7].iloc[i])]
            #find smallest value
            for l in range(len(df_sub)):
                smallM.append(np.sum(mask==df_sub[2].iloc[l]))
                indexM.append(df_sub[2].iloc[l])
            argM = np.argmin(smallM)
            delete_df.append(df_sub[df_sub[2]==indexM[argM]].index[0])
        delI = np.unique(delete_df)
        df = df.drop(delI)
        df = df.reset_index(drop=True)

        df = df.drop(delete_row)
        df = df.reset_index(drop=True)
        index_remove = list(np.where(df[4]==df[6])[0])
        df = df.drop(index_remove,axis='index')
        df = df.reset_index(drop=True)
        df_new = df.rename(columns={0: "old pos1", 1: "old pos2", 2: "map value", 3: "distance between pos", 4: "new pos1", 5: "node pos1", 6: "new pos2", 7: "node pos2"})
    return df_new


def edge_len(mask, df_pos,m):
    '''
    Calculate the length of each edge

    Parameters
    ----------
    mask : array
        mask of edges.
    df_pos : TYPE
        list of node positions.
    m : TYPE
        which edge is indexed in the mask array.

    Returns
    -------
    dist : float
        The distance of the edge.

    '''
    # calculate length for one edge
    edge_img = (mask==df_pos['map value'][m])*1
    (rows,cols) = np.nonzero(edge_img)
    dist = 1
    for (r,c) in zip(rows[:-1],cols[:-1]):
        if((edge_img[r,c+1]==1) or (edge_img[r,c-1]==1) or (edge_img[r+1,c]==1) or (edge_img[r-1,c]==1)):
            dist +=1
            edge_img[r-1:r+2,c-1:c+2]=0
        else:
            dist+= np.sqrt(2)
            edge_img[r-1:r+2,c-1:c+2]=0
    return dist


def make_graph_mask(imageAnnotated, imG, mask, df_pos, imageHysteresisCleaned):
    '''
    Parameters
    ----------
    imageAnnotated : array
        DESCRIPTION.
    imG : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    df_pos : TYPE
        DESCRIPTION.

    Returns
    -------
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.

    '''
    pos = np.transpose(np.where(imageAnnotated > 1))[:, ::-1]
    nodeNumber = imageAnnotated.max() - 1
    graph = nx.empty_graph(nodeNumber, nx.MultiGraph())

    ## Extension
    # Calculate the distance transform to get the width of each pixel
    # Invert the mask to calculate distance to the nearest background pixel
    #mask_inverted = (imageHysteresisCleaned == 0)

    imageHysteresisCleaned_padded = np.pad(imageHysteresisCleaned, 1, 'constant')
    distance_map = ndimage.distance_transform_edt(imageHysteresisCleaned_padded)
    pixel_widths = distance_map * 2
    pixel_widths_skeletonized = skimage.morphology.skeletonize(pixel_widths > 0)
    pixel_widths_skeletonized_value = pixel_widths * pixel_widths_skeletonized
    ##

    for m in range(len(df_pos)):
        node1,node2 = imageAnnotated[df_pos["new pos1"][m][0],df_pos["new pos1"][m][1]]-2, imageAnnotated[df_pos["new pos2"][m][0],df_pos["new pos2"][m][1]]-2

        #* Extension: Extract the mask for the current edge
        edge_mask = (mask == df_pos['map value'][m])*1
        # Calculate the width of each pixel in the edge
        pixel_widths_per_edge = pixel_widths_skeletonized_value * edge_mask
        pixel_weight_per_intensity = imG * edge_mask
        filamentIntensitySum = np.sum(imG * (mask == df_pos['map value'][m]) * 1)
        nodeDistance    = np.sqrt(
            (df_pos['new pos1'][m][0] - df_pos['new pos2'][m][0])**2 +
            (df_pos['new pos1'][m][1] - df_pos['new pos2'][m][1])**2
        )
        filamentLengthSum = max(nodeDistance, edge_len(mask, df_pos, m))
        minimumEdgeWeight = max(1e-9, filamentIntensitySum)
        edgeCapacity      = 1.0 * minimumEdgeWeight / filamentLengthSum
        edgeLength        = 1.0 * filamentLengthSum / minimumEdgeWeight
        edgeConnectivity  = 0

        mask_edge         = pixel_widths_per_edge > 0
        average_width     = np.mean(pixel_widths_per_edge[mask_edge])
        max_width         = np.max(pixel_widths_per_edge[mask_edge])
        min_width         = np.min(pixel_widths_per_edge[mask_edge])

        mask_edge_int      = pixel_weight_per_intensity > 0
        filamentIntensityAvg = np.mean(pixel_weight_per_intensity[mask_edge_int])


        graph.add_edge(node1, node2, edist=nodeDistance, fdist=filamentLengthSum, avg_weight = filamentIntensityAvg ,weight=minimumEdgeWeight, capa=edgeCapacity, lgth=edgeLength,avg_width=average_width, max_width=max_width, min_width=min_width, conn=edgeConnectivity)
    return graph,pos


def unify_graph(graph):
    '''
    Project graph to simple graph

    Parameters
    ----------
    graph : nx graph
        graph.

    Returns
    -------
    simpleGraph : nx simple graph
        graph containing the same values, now converted.

    '''
    simpleGraph = nx.empty_graph(graph.number_of_nodes())
    for node1, node2, property in graph.edges(data=True):
        edist = property['edist']
        fdist = property['fdist']
        weight = property['weight']
        capa = property['capa']
        lgth = property['lgth']
        conn = property['conn']

        #Add width properties
        avg_width= property['avg_width']
        max_width= property['max_width']
        min_width= property['min_width']

        avg_weight = property['avg_weight']


        if simpleGraph.has_edge(node1, node2):
            simpleGraph[node1][node2]['capa'] += capa
            if(simpleGraph[node1][node2]['lgth'] > lgth):
                simpleGraph[node1][node2]['lgth'] = lgth

            # Combine or update thickness properties
            simpleGraph[node1][node2]['avg_width'] = (simpleGraph[node1][node2]['avg_width'] + avg_width) / 2  # Averaging
            simpleGraph[node1][node2]['max_width'] = max(simpleGraph[node1][node2]['max_width'], max_width)    # Keep maximum
            simpleGraph[node1][node2]['min_width'] = min(simpleGraph[node1][node2]['min_width'], min_width)    # Keep minimum

            simpleGraph[node1][node2]['avg_weight'] = (simpleGraph[node1][node2]['avg_weight'] + avg_weight) / 2  # Averaging

        else:
            simpleGraph.add_edge(node1, node2, edist=edist, fdist=fdist, avg_weight=avg_weight, weight=weight, capa=capa, lgth=lgth, avg_width=avg_width, max_width=max_width, min_width=min_width, conn=conn)

    return simpleGraph


def test_connectivity(graph):
    '''
    function to remove if contain single node as connected component

    Parameters
    ----------
    graph : nx graph
        raw graph.

    Returns
    -------
    graph : nx graph
        filtered graph.

    '''
    c_list = [list(c) for c in nx.connected_components(graph)]
    for i in c_list:
        if(len(i)==1):
            log.debug("Removing isolated single-node component: %s", i)
            graph.remove_node(i[0])
    return graph



def dangling_edges(graph1):
    '''
    mark all edges as either dangling (1) or not (0)

    Parameters
    ----------
    graph1 : nx graph
        Original graph.

    Returns
    -------
    The graph with marked dangling edges.

    '''
    # mark all edges with dangling = 0
    for node1, node2, property in graph1.edges(data=True):
        graph1[node1][node2]['dangling'] = 0
        graph1[node1][node2]['filament dangling'] = 0


    node_degree1_list = [k for k,v in nx.degree(graph1) if ((v == 1) or (v == 3))]

    for node1,node2 in graph1.edges(node_degree1_list):
        graph1[node1][node2]['dangling'] = 1
    return(graph1)



def angle_between_edges(node1,node2,pos):
    '''
    Takes nodes from the linegraph as input, so nodes represents edges.
    From this can calculate the angle created between two edges

    Parameters
    ----------
    node1 : float
        first node, which represents edge.
    node2 : float
        Second node, which represents edge.
    pos : list
        list of position of nodes.

    Returns
    -------
    angle_deg : float
        The angle between two edges.

    '''

    same = np.intersect1d(node1,node2)
    index1 = np.where(node1==same)
    index2 = np.where(node2==same)

    if((index1[0]==1) and (index2[0]==1)):
        edgepos10 = pos[node1[0]]
        edgepos11 = pos[node1[1]]
        edgepos20 = pos[node2[0]]
        edgepos21 = pos[node2[1]]

    elif((index1[0]==0) and (index2[0]==1)):
        edgepos10 = pos[node1[0]]
        edgepos11 = pos[node1[1]]
        edgepos20 = pos[node2[1]]
        edgepos21 = pos[node2[0]]

    elif((index1[0]==0) and (index2[0]==0)):
        edgepos10 = pos[node1[0]]
        edgepos11 = pos[node1[1]]
        edgepos20 = pos[node2[0]]
        edgepos21 = pos[node2[1]]

    elif((index1[0]==1) and (index2[0]==0)):
        edgepos10 = pos[node1[0]]
        edgepos11 = pos[node1[1]]
        edgepos20 = pos[node2[1]]
        edgepos21 = pos[node2[0]]


    vec1 = (edgepos11[0]-edgepos10[0], edgepos11[1]-edgepos10[1])
    vec2 = (edgepos21[0]-edgepos20[0], edgepos21[1]-edgepos20[1])
    unit_vector_1 = vec1 / np. linalg. norm(vec1)
    unit_vector_2 = vec2 / np. linalg. norm(vec2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if(dot_product>1):
        dot_product=1
    elif(dot_product<-1):
        dot_product=-1
    angle = np.arccos(dot_product)

    angle_deg = angle*180./np.pi
    return angle_deg

def lG_edgeVal(lg1,graph1,pos):
    '''
    takes the linegraph and populate with angles, as well as marking all dangling nodes

    Parameters
    ----------
    lg1 : nx graph
        linegraph.
    graph1 : nx graph
        Original graph.
    pos : list
        List of positions of nodes.

    Returns
    -------
    lg1 : nx graph
        Linegraph populated with angle values of the nodes aka edges of the original graph.

    '''

    for node1, node2, property in lg1.edges(data=True):
        # calculate the angle between these two edges from the original graph
        # this is done by the positions in image
        lg1[node1][node2]['angle'] = angle_between_edges(node1,node2,pos)

        lg1[node1][node2]['intensity'] = np.abs(graph1[node1[0]][node1[1]]['capa'] - graph1[node2[0]][node2[1]]['capa'])/min(graph1[node1[0]][node1[1]]['capa'], graph1[node2[0]][node2[1]]['capa'])
        lg1.nodes[node1]['dangling'] =  graph1[node1[0]][node1[1]]['dangling']
        lg1.nodes[node2]['dangling'] =  graph1[node2[0]][node2[1]]['dangling']
    return lg1


###############################################################################
#
# tracing step
#
###############################################################################

def all_angles(lg,pos):
    '''
    takes the linegraph and nodepositions in and reated a dataframe with nodepos and belonging angle

    Parameters
    ----------
    lg : array
        linegraph of graph.
    pos : list
        node positions.

    Returns
    -------
    df_new : list
        dataframe of nodepositions together with angle.

    '''
    all_anglesl = [list(c) for c in lg.edges(data='angle')]
    pd_al = pd.DataFrame(all_anglesl)
    df_new = pd_al.rename(columns={0: "pos1", 1: "pos2", 2: "angle"})
    return df_new


def min_angle(graph,imgBl,pos,angle):
    '''
    calculate minimum angle to use in constrained DFS

    Parameters
    ----------
    graph : nx graph
        input graph.
    imgBl : array
        image with the added nodes from VW algorithm.
    pos : list
        list of node positions.
    angle : float
        user input of angle.

    Returns
    -------
    min_angle_val : float
        Minimum angle allowed for the constrained DFS.

    '''
    graphD = dangling_edges(graph)
    lgG = nx.line_graph(graph)
    lg1 = lG_edgeVal(lgG,graphD,pos)

    # define angle list, find all coordinates in blank image and then find the
    # overlap between pos and these. then find the angles between these coordinates
    # return this smallest value
    angles_list = [angle]
    (rows,cols) = np.nonzero((imgBl>0)*1)
    posBl = np.vstack((cols,rows)).T

    ind_pos = np.zeros(len(posBl))
    ind_posD = []
    for i in range(len(posBl)):
        ind_pos[i] = np.argmin(abs(posBl[i][0]-pos[:,0])+abs(posBl[i][1]-pos[:,1]))
        if(abs(posBl[i][0]-pos[int(ind_pos[i]),0])+abs(posBl[i][1]-pos[int(ind_pos[i]),1])>1):
            ind_posD.append(i)
    ind_posN = np.delete(ind_pos,ind_posD)

    angles = [list(c) for c in lg1.edges(data='angle')]

    for j in range(len(ind_posN)):
        for i in range(len(angles)):
            if((ind_posN[j] in angles[i][0:1][0]) and (ind_posN[j] in angles[i][1:2][0])):
                np.asarray(angles[i][0:1]).flatten()
                angles_list.append(angles[i][2])
    min_angle_val = min(angles_list) - 1
    while(min_angle_val < 90):
        angles_list.remove(min(angles_list))
        min_angle_val = min(angles_list) - 1
    return min_angle_val



def dfs_constrained(graph_s, lgG_V, imgBl, pos, angle, overlap_allowed,
                    intensity_tolerance, thickness_tolerance,
                    intensity_coeff=3, thickness_coeff=3, angle_coeff=5, score_threshold=8, percentile_intensity = 30, percentile_thickness = 50,
                    apply_angle_penalty=True, either_constraint_coeff=0):
    '''
    constrained DFS function, to define and trace individual filaments

    Parameters
    ----------
    graph_s : nx graph
        graph from image data.
    lgG_V : nx graph
        linegraph from graph from image data.
    imgBl : array
        image with the added nodes from VW algorithm.
    pos : list
        list of node positions.
    angle : float
        minimum angle allowed.
    overlap_allowed : float
        minimum overlap allowed.
    intensity_coeff : int, optional
        Coefficient weighting the intensity constraint.
    thickness_coeff : int, optional
        Coefficient weighting the thickness constraint.
    angle_coeff : int, optional
        Coefficient weighting the angle constraint.
    score_threshold : int, optional
        Score threshold for accepting an edge.
    intensity_tolerance : float or None, optional
        If not None, use this intensity tolerance (overriding dynamic computation).
    thickness_tolerance : float or None, optional
        If not None, use this thickness tolerance (overriding dynamic computation).

    Returns
    -------
    graphTagF : nx graph
        graph with tagged filament values.

    '''

    # Save the caller-supplied tolerance values so that each connected component
    # gets an independent dynamic computation (or reuses the fixed value if set).
    _intensity_tolerance_init = intensity_tolerance
    _thickness_tolerance_init = thickness_tolerance

    graphTag = graph_s.copy()
    graphTag = dangling_edges(graphTag)
    c_list = [list(c) for c in nx.connected_components(graph_s)]
    tag = 0
    to_add = []

    while c_list:
        # Reset tolerances so each component computes its own dynamic threshold.
        intensity_tolerance = _intensity_tolerance_init
        thickness_tolerance = _thickness_tolerance_init
        log.debug("Connected components remaining: %s", c_list)
        c = c_list.pop(0)
        G = graph_s.subgraph(sorted(c)).copy()
        log.debug("Processing subgraph with nodes: %s", c)

        angle_min_val = min_angle(G, imgBl, pos, angle)
        al_pd = all_angles(lgG_V, pos)

        # ------------------------------------------------------------------------------------------------------------------------
        #* Calculate dynamically the intensity and thickness tolerance using the percentile (or MAD, IQR, Mean ...)
        # Test phase, but improvement. Compare to the static and absolute thresholds.
        intensity_ratios = []
        thickness_ratios = []

        # Loop through all edges in the connected component and get the edge. Then use the neighbors to get the neighbor with either node as a sharing node.
        for edge in G.edges(data=True):
            node1, node2, edge_data = edge
            neighbors_node1 = list(G.neighbors(node1))
            neighbors_node2 = list(G.neighbors(node2))

            # Find edges sharing node1 within G
            for neighbor in neighbors_node1:
                if neighbor != node2 and G.has_edge(node1, neighbor):
                    neighbor_edge_data = G[node1][neighbor]
                    # Intensity ratio normalization by average
                    avg_intensity = (edge_data['avg_weight'] + neighbor_edge_data['avg_weight']) / 2
                    if avg_intensity > 0:  # Avoid division by zero
                        intensity_ratio = abs(edge_data['avg_weight'] - neighbor_edge_data['avg_weight']) / avg_intensity
                        intensity_ratios.append(intensity_ratio)

                    # Thickness ratio normalization by average
                    avg_thickness = (edge_data['avg_width'] + neighbor_edge_data['avg_width']) / 2
                    if avg_thickness > 0:
                        thickness_ratio = abs(edge_data['avg_width'] - neighbor_edge_data['avg_width']) / avg_thickness
                        thickness_ratios.append(thickness_ratio)

            # Find edges sharing node2 within G
            for neighbor in neighbors_node2:
                if neighbor != node1 and G.has_edge(node2, neighbor):
                    neighbor_edge_data = G[node2][neighbor]
                    # Intensity ratio normalization by average
                    avg_intensity = (edge_data['avg_weight'] + neighbor_edge_data['avg_weight']) / 2
                    if avg_intensity > 0:
                        intensity_ratio = abs(edge_data['avg_weight'] - neighbor_edge_data['avg_weight']) / avg_intensity
                        intensity_ratios.append(intensity_ratio)

                    # Thickness ratio normalization by average
                    avg_thickness = (edge_data['avg_width'] + neighbor_edge_data['avg_width']) / 2
                    if avg_thickness > 0:
                        thickness_ratio = abs(edge_data['avg_width'] - neighbor_edge_data['avg_width']) / avg_thickness
                        thickness_ratios.append(thickness_ratio)

        intensity_ratios = np.array(intensity_ratios)
        thickness_ratios = np.array(thickness_ratios)

        # Calculate thresholds for this connected component using percentiles or MAD
        # If the user provided intensity_tolerance (i.e. not None), use it.
        if intensity_tolerance is None:
            if len(intensity_ratios) > 0:
                intensity_tolerance = np.percentile(intensity_ratios, percentile_intensity)
            else:
                intensity_tolerance = 0
            log.debug("Dynamic intensity_tolerance = %.4f", intensity_tolerance)
        else:
            log.debug("Using fixed intensity_tolerance = %.4f", intensity_tolerance)

        if thickness_tolerance is None:
            if len(thickness_ratios) > 0:
                thickness_tolerance = np.percentile(thickness_ratios, percentile_thickness)
            else:
                thickness_tolerance = 0
            log.debug("Dynamic thickness_tolerance = %.4f", thickness_tolerance)
        else:
            log.debug("Using fixed thickness_tolerance = %.4f", thickness_tolerance)

        log.debug("intensity_tolerance=%.4f  thickness_tolerance=%.4f",
                  intensity_tolerance, thickness_tolerance)

        # ------------------------------------------------------------------------------------------------------------------------

        all_stacks = []
        path_full = []
        path_keep = []

        dangling_nodeS = [k for k, v in nx.degree(G) if ((v == 1))]
        dangling_nodeE = [k for k, v in nx.degree(G) if ((v == 3))]
        dangling_node4 = [k for k, v in nx.degree(G) if ((v == 4))]

        dangling_nodeS.extend(dangling_nodeE)
        dangling_nodeS.extend(dangling_node4)
        dangling_nodeSC = dangling_nodeS.copy()

        df_I = pd.DataFrame(graph_s.edges(data='capa'))
        df_I[3] = np.asarray(list(graph_s.edges(data='fdist')))[:, 2]
        df_I[4] = np.asarray(list(graph_s.edges(data='avg_width')))[:, 2]
        df_I[5] = np.asarray(list(graph_s.edges(data='avg_weight')))[:, 2]
        df_I = df_I.rename(columns={0: "node1", 1: "node2", 2: "capa", 3: "fdist", 4:"avg_width", 5:"avg_weight"})
        # need to add a check in case the smallest node value is not node1 from node2
        for i in range(len(df_I)):
            if df_I['node1'].iloc[i] > df_I['node2'].iloc[i]:
                #switch them around
                nod1 = df_I['node1'].iloc[i]
                nod2 = df_I['node2'].iloc[i]
                df_I.loc[i, 'node1'] = nod2
                df_I.loc[i, 'node2'] = nod1


        #* Extension to consider for closed shapes: First check if there are dangling nodes.
        is_cycle = False
        if len(dangling_nodeSC) == 0:
            log.debug("No dangling nodes — treating subgraph as a closed shape.")
            closed_shape_nodes = [k for k, v in nx.degree(G) if v == 2]
            dangling_nodeS  = closed_shape_nodes[::1]
            dangling_nodeSC = dangling_nodeS.copy()
            is_cycle = True

        if len(dangling_nodeSC) == 2:
            node1, node2 = dangling_nodeSC
            if graph_s.has_edge(node1, node2):
                log.debug("Only one effective dangling node — adding degree-2 nodes for full traversal.")
                second_degree_nodes = [k for k, v in nx.degree(G) if v == 2]
                additional_nodes    = second_degree_nodes[::2]
                dangling_nodeS.extend(
                    [n for n in additional_nodes if n not in dangling_nodeS]
                )
                dangling_nodeSC = dangling_nodeS.copy()

        starting_node = dangling_nodeSC[0]  # Keep track of the starting node


        while dangling_nodeSC:
            log.debug("Dangling nodes remaining: %s", dangling_nodeSC)
            dangling_del = []
            source = dangling_nodeSC.pop(0)
            nodes       = [source]
            visited     = set()
            depth_limit = len(G)
            last_intensity = None
            last_thickness = None

            for start in nodes:
                if start in visited:
                    log.debug("Start node %s already visited — skipping.", start)
                    continue
                visited.add(start)
                stack = [(start, depth_limit, iter(G[start]))]
                log.debug("DFS from starting node %s", starting_node)

                while stack:
                    parent, depth_now, children = stack[-1]
                    log.debug("DFS parent=%s depth=%s", parent, depth_now)
                    try:
                        child = next(children)
                        log.debug("Evaluating child=%s", child)
                        if child not in visited:
                            current_intensity = graph_s[parent][child]['avg_weight']
                            current_thickness = graph_s[parent][child]['avg_width']
                            log.debug("parent=%s child=%s intensity=%.4f thickness=%.4f",
                                      parent, child, current_intensity, current_thickness)

                            total_score = 0
                            intensity_ok = False
                            thickness_ok = False
                            if last_intensity is not None:
                                avg_intensity   = (current_intensity + last_intensity) / 2
                                intensity_ratio = abs(current_intensity - last_intensity) / avg_intensity
                                if intensity_ratio <= intensity_tolerance:
                                    total_score += intensity_coeff
                                    intensity_ok = True
                                log.debug("intensity_ratio=%.4f tolerance=%.4f score=%d",
                                          intensity_ratio, intensity_tolerance, total_score)
                            else:
                                total_score += score_threshold

                            if last_thickness is not None:
                                avg_thickness   = (current_thickness + last_thickness) / 2
                                thickness_ratio = abs(current_thickness - last_thickness) / avg_thickness
                                if thickness_ratio <= thickness_tolerance:
                                    total_score += thickness_coeff
                                    thickness_ok = True
                                log.debug("thickness_ratio=%.4f tolerance=%.4f score=%d",
                                          thickness_ratio, thickness_tolerance, total_score)
                            else:
                                total_score += score_threshold

                            # Demonstration override: if EITHER the intensity or
                            # thickness constraint is satisfied, add a large bonus
                            # so the edge is kept regardless of the angle term.
                            # Off (no effect) when either_constraint_coeff == 0.
                            if either_constraint_coeff > 0 and (intensity_ok or thickness_ok):
                                total_score += either_constraint_coeff

                            if len(path_keep) >= 1:
                                entry = al_pd['angle'].loc[
                                    ((al_pd['pos1'] == (min(parent,child),max(parent,child))) | (al_pd['pos1'] == path_keep[-1])) &
                                    ((al_pd['pos2'] == (min(parent,child),max(parent,child))) | (al_pd['pos2'] == path_keep[-1]))
                                ]
                                log.debug("Angle entry: %s", entry.values)
                                if entry.size > 0:
                                    if entry.values >= angle_min_val:
                                        total_score += angle_coeff
                                        if entry.values >= 175:
                                            total_score += angle_coeff  # near-straight bonus
                                    elif entry.values * 1.4 < angle_min_val:
                                        if apply_angle_penalty:
                                            total_score -= angle_coeff
                                    elif entry.values < angle_min_val:
                                        if len(path_full) == 0:
                                            if nx.degree(G, parent) in {2, 4}:
                                                dangling_del.append(child)
                                                path_full.append(path_keep.copy())
                                        elif (path_full[-1] != path_keep) and (nx.degree(G, parent) in {2, 4}):
                                            path_full.append(path_keep.copy())
                                            dangling_del.append(child)
                            else:
                                total_score += score_threshold

                            if total_score >= score_threshold:
                                path_keep.append((min(parent, child), max(parent, child)))
                                visited.add(child)
                                last_intensity = current_intensity
                                last_thickness = current_thickness
                                if depth_now > 1:
                                    stack.append((child, depth_now - 1, iter(G[child])))
                                if child in dangling_nodeS:
                                    all_stacks.append(stack.copy())
                                    path_full.append(path_keep.copy())
                                    dangling_del.append(child)

                        elif (is_cycle and child == starting_node and depth_now == 1):
                            log.debug("Closing cycle at starting node %s.", starting_node)
                            current_intensity = graph_s[parent][child]['capa']
                            current_thickness = graph_s[parent][child]['avg_width']

                            total_score = 0
                            intensity_ok = False
                            thickness_ok = False
                            if last_intensity is not None:
                                avg_intensity   = (current_intensity + last_intensity) / 2
                                intensity_ratio = abs(current_intensity - last_intensity) / avg_intensity
                                if intensity_ratio <= intensity_tolerance:
                                    total_score += intensity_coeff
                                    intensity_ok = True
                            else:
                                total_score += score_threshold

                            if last_thickness is not None:
                                avg_thickness   = (current_thickness + last_thickness) / 2
                                thickness_ratio = abs(current_thickness - last_thickness) / avg_thickness
                                if thickness_ratio <= thickness_tolerance:
                                    total_score += thickness_coeff
                                    thickness_ok = True
                            else:
                                total_score += score_threshold

                            if either_constraint_coeff > 0 and (intensity_ok or thickness_ok):
                                total_score += either_constraint_coeff

                            entry = al_pd['angle'].loc[
                                ((al_pd['pos1'] == (min(parent,child),max(parent,child))) | (al_pd['pos1'] == path_keep[-1])) &
                                ((al_pd['pos2'] == (min(parent,child),max(parent,child))) | (al_pd['pos2'] == path_keep[-1]))
                            ]
                            if entry.size > 0:
                                if entry.values >= angle_min_val:
                                    total_score += angle_coeff
                                    if entry.values >= 175:
                                        total_score += angle_coeff
                                elif entry.values * 1.4 < angle_min_val:
                                    if apply_angle_penalty:
                                        total_score -= angle_coeff
                                elif entry.values < angle_min_val:
                                    if len(path_full) == 0:
                                        if nx.degree(G, parent) in {2, 4}:
                                            dangling_del.append(child)
                                            path_full.append(path_keep.copy())
                                    elif (path_full[-1] != path_keep) and (nx.degree(G, parent) in {2, 4}):
                                        path_full.append(path_keep.copy())
                                        dangling_del.append(child)
                            if total_score >= score_threshold:
                                log.debug("Closing cycle — adding edge (%s, %s).", parent, child)
                                path_keep.append((min(parent, child), max(parent, child)))
                                visited.add(child)
                                last_intensity = current_intensity
                                last_thickness = current_thickness
                                if depth_now > 1:
                                    stack.append((child, depth_now - 1, iter(G[child])))
                                if child in dangling_nodeS:
                                    all_stacks.append(stack.copy())
                                    path_full.append(path_keep.copy())
                                    dangling_del.append(child)

                            continue


                        elif (
                            (child in np.asarray([item for sublist in path_keep for item in sublist])) and
                            (child not in path_keep[-1]) and
                            ((child in dangling_nodeE) or (child in dangling_node4)) and
                            (1 != path_keep.count((min(child, parent), max(child, parent))))
                        ):
                            log.debug("Re-visiting complex-structure node %s via parent %s.", child, parent)
                            path_keep.append((min(parent, child), max(parent, child)))
                            stack.append((child, depth_now - 1, iter(G[child])))
                            all_stacks.append(stack.copy())
                            if child in dangling_nodeE:
                                path_keep1 = list(set(path_keep) - set(path_full[-1]))
                                path_full.append(path_keep1)
                            path_full.append(path_keep.copy())
                            dangling_del.append(child)

                        else:
                            log.debug("Node %s already visited — edge not added.", child)

                    except StopIteration:
                        log.debug("No more children for parent %s — backtracking.", parent)
                        stack.pop()
                        if path_keep:
                            path_keep.pop()


        log.debug("path_keep=%s", path_keep)
        log.debug("path_full (before extend)=%s", path_full)
        path_full.extend([element] for element in list(graph_s.edges(sorted(c))))
        log.debug("path_full (after extend)=%s", path_full)
        len_path = np.zeros(len(path_full))

        for i in range(len(path_full)):
            path_c = path_full[i]
            iLl = np.zeros(len(path_c))
            for k in range(len(path_c)):
                iLl[k] = float(df_I['fdist'][(df_I['node1'] == path_c[k][0]) & (df_I['node2'] == path_c[k][1])].iloc[0])
            len_fil = np.sum(iLl)
            len_path[i] = len_fil

        idx = np.argsort(len_path)
        path_keep = []
        full_len_path = len_path.copy()
        path_keep.append(path_full[idx[-1]])

        flat_list = np.asarray([item for sublist in path_keep for item in sublist])
        G_list = np.asarray(list(graph_s.edges(sorted(c))))
        not_covered = G_list[~(G_list[:, None] == flat_list).all(-1).any(-1)]

        while len(not_covered) > 0:
            len_path = np.zeros(len(path_full))
            for i in range(len(path_full)):
                path_c = path_full[i]
                iLl = np.zeros(len(path_c))
                len_fil = 0
                for k in range(len(path_c)):
                    if np.isin(np.asarray(path_c[k]), np.asarray(not_covered)).all():
                        iLl[k] = float(df_I['fdist'][(df_I['node1'] == path_c[k][0]) & (df_I['node2'] == path_c[k][1])].iloc[0])
                len_fil = np.sum(iLl)
                len_path[i] = len_fil

            idx = np.argsort(len_path)

            flat_list = [item for sublist in path_keep for item in sublist]
            switch = 1
            while switch:
                if not ((np.sum(np.isin(np.asarray(flat_list), np.asarray(path_full[idx[-1]])), axis=1) == 2).any()):
                    switch = 0
                elif ((np.sum(np.isin(np.asarray(flat_list), np.asarray(path_full[idx[-1]][0])), axis=1) == 2).any() or
                      (np.sum(np.isin(np.asarray(flat_list), np.asarray(path_full[idx[-1]][-1])), axis=1) == 2).any()):
                    idx = np.delete(idx, -1)
                else:
                    overlap_ind = np.asarray(np.sum(np.isin(np.asarray(flat_list), np.asarray(path_full[idx[-1]])), axis=1) == 2).nonzero()[0]
                    overlap = [flat_list[i] for i in overlap_ind]
                    iLo = np.zeros(len(overlap))
                    for k in range(len(overlap)):
                        iLo[k] = float(df_I['fdist'][(df_I['node1'] == overlap[k][0]) & (df_I['node2'] == overlap[k][1])].iloc[0])
                    len_overlap = np.sum(iLo)
                    if len_overlap > full_len_path[idx[-1]] / overlap_allowed:
                        idx = np.delete(idx, -1)
                    else:
                        switch = 0

            path_keep.append(path_full[idx[-1]])

            flat_list = np.asarray([item for sublist in path_keep for item in sublist])
            G_list = np.asarray(list(graph_s.edges(sorted(c))))
            not_covered = G_list[~(G_list[:, None] == flat_list).all(-1).any(-1)]

        for l in range(len(path_keep)):
            edges_to_remove = path_keep[l]
            overlap = []
            for i in range(l + 1, len(path_keep)):
                if (np.sum(np.isin(np.asarray(path_keep[l]), np.asarray(path_keep[i])), axis=1) == 2).any():
                    overlap_val = [x for x in path_keep[l] if x in path_keep[i]]
                    overlap.extend(overlap_val)
            if len(overlap) > 0:
                index_o = np.zeros(len(path_keep[l]), dtype=bool)
                for m in range(len(overlap)):
                    curI = np.where(np.sum(np.isin(np.asarray(path_keep[l]), np.asarray(overlap[m])), axis=1) == 2)[0]
                    index_o[curI] = 1
                overlap = [val for is_good, val in zip(index_o, path_keep[l]) if is_good]
                edges_to_remove = [val for is_good, val in zip(~index_o, path_keep[l]) if is_good]

                g_sub = graphTag.edge_subgraph(overlap).copy()
                for z in range(len(overlap)):
                    g_sub[overlap[z][0]][overlap[z][1]]['filament'] = tag

                to_add.extend(list(g_sub.edges(data=True)))
            if len(edges_to_remove) > 0:
                for m in range(len(edges_to_remove)):
                    # The same edge can be scheduled for removal by more than one
                    # kept path when paths overlap heavily (e.g. under the forced
                    # either-constraint merge), so guard against removing it twice.
                    if G.has_edge(edges_to_remove[m][0], edges_to_remove[m][1]):
                        G.remove_edge(edges_to_remove[m][0], edges_to_remove[m][1])
                    graphTag[edges_to_remove[m][0]][edges_to_remove[m][1]]['filament'] = tag
                graphTag[path_keep[l][0][0]][path_keep[l][0][1]]['filament dangling'] = 1
                graphTag[path_keep[l][-1][0]][path_keep[l][-1][1]]['filament dangling'] = 1

                tag += 1

        graphTagF = nx.MultiGraph(graphTag.copy())
        graphTagF.add_edges_from(to_add)

    return graphTagF





###############################################################################
#
# drawing functions
#
###############################################################################

def draw_graph(image,graph,pos,title):
    '''
    simple drawing function

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image, cmap='gray')
    nx.draw(graph, pos, node_size=30, with_labels=True,
            edge_color='red', font_color='white', font_size=14, alpha=0.5)
    return fig

def draw_graph_filament_nocolor(image,graph,pos,title,value):
    '''
    drawing function

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.

    Returns
    -------
    saved image

    '''
    edges, values = zip(*nx.get_edge_attributes(graph, value).items())

    fig = plt.figure(figsize=(8, 10))
    plt.title(title)

    height, width = image.shape[:2]
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.gca().set_aspect('equal')

    vmax = max(values) + 1
    if vmax % 2 == 0:
        vmax += 1
    cmap = plt.get_cmap('tab20', vmax)

    nx.draw(graph, pos, edge_cmap=cmap, edge_color=values,
            node_size=0.7, width=5, alpha=0.5)

    for u, v, d in graph.edges(data=True):
        mid_pos = (np.array(pos[u]) + np.array(pos[v])) / 2
        plt.text(mid_pos[0], mid_pos[1], str(d[value]),
                 color='black', fontsize=14, ha='center', va='center')

    plt.gca().set_axis_off()
    plt.tight_layout()
    return fig



###############################################################################
#
# Quantification functions
#
###############################################################################

def adjust_filament_tags(graph):
    """
    Adjusts filament tags in the graph to ensure they start from 1 and do not include 0.

    Parameters:
    - graph: NetworkX graph with 'filament' attribute on edges.

    Returns:
    - new_graph: NetworkX graph with adjusted filament tags.
    - tag_mapping: Dictionary mapping old filament tags to new filament tags.
    """
    # Get all unique filament tags
    filament_tags = set()
    for u, v, data in graph.edges(data=True):
        filament_tag = data.get('filament')
        if filament_tag is not None:
            filament_tags.add(filament_tag)

    # Create a mapping from old tags to new tags, starting from 1
    sorted_tags = sorted(filament_tags)
    tag_mapping = {old_tag: new_tag for new_tag, old_tag in enumerate(sorted_tags, start=1)}

    # Create a copy of the graph to avoid modifying the original
    new_graph = graph.copy()

    # Update filament tags in the new graph
    for u, v, data in new_graph.edges(data=True):
        filament_tag = data.get('filament')
        if filament_tag is not None:
            new_tag = tag_mapping[filament_tag]
            data['filament'] = new_tag

    return new_graph, tag_mapping



def prepare_ground_truth_layers(ground_truth_folder):
    """
    Load per-filament ground-truth layer masks from a folder.

    Only files named ``label<n>`` (e.g. ``label1.png``, ``label2.png``) with a
    lossless raster extension (.png/.tif/.tiff) are loaded, ordered by their
    numeric index ``<n>``. ``label*`` is the canonical mask name in this
    project's ground-truth folders; reference material (source scans, overlays
    named ``layer*``, ``merged_labeled_image.png``, skeleton previews, etc.) is
    kept in a ``source/`` subfolder and is intentionally not matched.

    This deliberately ignores other images that might share the folder and
    would otherwise be picked up by a blanket extension filter:
      - overlays named ``layer*`` (often RGBA, filament drawn dark-on-white, so
        a ``> 0`` binarization would mark the whole frame as foreground),
      - source scans (often *.jpeg) at a different resolution,
      - composite previews such as ``merged_labeled_image.png``,
      - skeleton previews such as ``skeleton_label1.png``.
    Because those have inconsistent shapes/semantics, including them used to
    break the downstream ``np.stack(...)`` in the caller. Restricting to the
    canonical ``label`` masks keeps every returned array the same shape. JPEG is
    excluded on purpose: it is lossy and unsuitable for label masks.

    Parameters:
    - ground_truth_folder: Path to the folder containing ground truth masks.

    Returns:
    - ground_truth_layers: List of binary (0/1) 2D numpy arrays, one per
      filament layer, ordered by index.
    """
    import os
    import re
    from skimage.io import imread

    # Anchored match: the *whole* filename must be label<digits> with a lossless
    # extension. Anchoring (^...$) is what excludes "skeleton_label1.png",
    # "merged_labeled_image.png", "img1.png", and any "layer*" overlay.
    pattern = re.compile(r'^label(\d+)\.(?:png|tif|tiff)$', re.IGNORECASE)

    matches = []
    for filename in os.listdir(ground_truth_folder):
        m = pattern.match(filename)
        if m:
            matches.append((int(m.group(1)), filename))

    if not matches:
        raise ValueError(
            f"No ground-truth layer masks found in '{ground_truth_folder}'. "
            "Expected files named like 'label1.png', 'label2.png', … "
            "(.png/.tif/.tiff)."
        )

    # Sort by the numeric index so layer2 comes before layer10 (lexicographic
    # sorting would order them 1, 10, 2, ...).
    matches.sort(key=lambda t: t[0])

    ground_truth_layers = []
    for _, filename in matches:
        layer = imread(os.path.join(ground_truth_folder, filename), as_gray=True)
        # Convert to binary mask
        layer_binary = (layer > 0).astype(np.uint8)
        ground_truth_layers.append(layer_binary)
    return ground_truth_layers


def get_line_pixels(coord_u, coord_v):
    """
    Get the pixel coordinates along the line between two points using Bresenham's algorithm.

    Parameters:
    - coord_u: Tuple (row, col) of the first node.
    - coord_v: Tuple (row, col) of the second node.

    Returns:
    - line_coords: Set of (row, col) coordinates along the line.
    """
    from skimage.draw import line
    row_u, col_u = coord_u
    row_v, col_v = coord_v
    rr, cc = line(row_u, col_u, row_v, col_v)
    return set(zip(rr, cc))


def extract_filament_coordinates_from_graph(graph, pos_dict):
    """
    Extracts coordinates of pixels belonging to each filament in the graph.

    Parameters:
    - graph: NetworkX graph with 'filament' attribute on edges.
    - pos_dict: Dictionary mapping node indices to their positions (row, col).

    Returns:
    - filament_coords_dict: Dictionary mapping filament tags to sets of (row, col) coordinates.
    """
    filament_coords_dict = {}
    for u, v, data in graph.edges(data=True):
        filament_tag = data.get('filament')
        if filament_tag is not None:
            # Get the coordinates of the edge
            coord_u = pos_dict[u]
            coord_v = pos_dict[v]
            # Get the pixels along the edge
            edge_coords = get_line_pixels(coord_u, coord_v)
            if filament_tag not in filament_coords_dict:
                filament_coords_dict[filament_tag] = set()
            filament_coords_dict[filament_tag].update(edge_coords)
    return filament_coords_dict


def create_predicted_label_image(filament_coords_dict, image_shape):
    """
    Creates a predicted label image from filament coordinates.

    Parameters:
    - filament_coords_dict: Dictionary mapping filament tags to sets of (row, col) coordinates.
    - image_shape: Tuple (height, width) of the image.

    Returns:
    - predicted_labels: 2D numpy array with pixel labels.
    """
    predicted_labels = np.zeros(image_shape, dtype=int)
    for filament_tag, coords in filament_coords_dict.items():
        for (row, col) in coords:
            predicted_labels[row, col] = filament_tag
    return predicted_labels


def relabel_ground_truth_layers(ground_truth_layers):
    """
    Relabels connected components in ground truth layers to ensure unique labels across all layers.
    Also visualizes each labeled layer with label numbers on each connected filament.

    Parameters:
    - ground_truth_layers: List of 2D numpy arrays (binary masks)

    Returns:
    - ground_truth_labels: 2D numpy array with unique labels for each filament
    - gt_filament_ids: List of unique ground truth filament IDs
    """
    ground_truth_labels = np.zeros_like(ground_truth_layers[0], dtype=int)
    current_label = 1  # Start labeling from 1
    gt_filament_ids = []

    for i, layer in enumerate(ground_truth_layers):
        # Label connected components in the layer
        labeled_layer = label(layer > 0, connectivity=2)
        unique_labels = np.unique(labeled_layer)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        num_labels = len(unique_labels)

        # Offset labels to ensure uniqueness
        labeled_layer[labeled_layer > 0] += current_label - 1

        # Update the combined ground truth labels
        ground_truth_labels += labeled_layer

        # Record filament IDs
        gt_filament_ids.extend(range(current_label, current_label + num_labels))

        # Update the current label for the next layer
        current_label += num_labels

    return ground_truth_labels, gt_filament_ids



def compute_confusion_matrix_multi_layer(predicted_labels, ground_truth_labels, predicted_filament_ids, gt_filament_ids):
    """
    Computes the confusion matrix between predicted filaments and ground truth filaments.

    Parameters:
    - predicted_labels: 2D numpy array with predicted filament labels
    - ground_truth_labels: 2D numpy array with ground truth filament labels
    - predicted_filament_ids: List of predicted filament IDs
    - gt_filament_ids: List of ground truth filament IDs

    Returns:
    - confusion_matrix: 2D numpy array where entry [i, j] is the number of pixels
                        where predicted filament i overlaps with ground truth filament j
    """
    # Define the dimensions of the confusion matrix
    n_pred = len(predicted_filament_ids)
    n_gt = len(gt_filament_ids)

    # Create label maps
    # Map each filament tag/label to a unique index as a dictionary
    pred_label_map = {label: idx for idx, label in enumerate(predicted_filament_ids)}
    gt_label_map = {label: idx for idx, label in enumerate(gt_filament_ids)}

    # Flatten the label imagesm by stacking the rows sequentially
    # Avoids nested loops for efficient iteration
    pred_flat = predicted_labels.flatten()
    gt_flat = ground_truth_labels.flatten()

    # Exclude background pixels and ensures that only relevant pixels are considered
    # Creates a mask that shows pixels if they are present in either the predicted or gtround_truth image
    mask = (pred_flat > 0) | (gt_flat > 0)
    #Filters the image to include only the pixels where the mask is True
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]

    # Map labels to indices using a list comprehension
    # Iterates trhough pred/gt_flat. If label is present in the respective array.
    # Then we get the index of the label in the dict that we created before. And store the indices.
    # If the label is not present then we use -1 as an index to filter it out for later processes.
    pred_indices = np.array([pred_label_map.get(l, -1) for l in pred_flat])
    gt_indices = np.array([gt_label_map.get(l, -1) for l in gt_flat])

    # Remove entries with -1 (background or unmapped labels)
    valid_mask = (pred_indices >= 0) & (gt_indices >= 0)
    pred_indices = pred_indices[valid_mask]
    gt_indices = gt_indices[valid_mask]

    # Create confusion matrix that quantifies the pixel-wise overlaps between each predicted filament and each ground truth filament
    # creates an array of 1s, the count 1 represents the number of overlaps
    data = np.ones_like(pred_indices)
    confusion_matrix = coo_matrix(
        # Tuple of arrays:
        # each pair (pred_indices[k], gt_indices[k]) corresponds to a pixel where the k-th predicted filament overlaps with the k-th ground truth filament.
        (data, (pred_indices, gt_indices)),
        # defines the dimensions: rows - predicted filaments and columns - ground truth filaments
        shape=(n_pred, n_gt),
    # converts the matrix into a dense numpy array for easier manipulation
    ).toarray()

    # Verify that all labels are mapped
    pred_unmapped = set(pred_flat) - set(pred_label_map.keys())
    gt_unmapped = set(gt_flat) - set(gt_label_map.keys())

    if pred_unmapped:
        log.debug("Unmapped predicted labels: %s", pred_unmapped)
    if gt_unmapped:
        log.debug("Unmapped ground truth labels: %s", gt_unmapped)


    # Utilizing sparse matrices (coo_matrix) helps mitigate memory issues by not storing the entire matrix in memory.
    # The use of NumPy's vectorized operations (like flatten and boolean masking) ensures that the function runs efficiently without explicit Python loops.
    return confusion_matrix


def match_filaments(confusion_matrix, predicted_labels, ground_truth_labels, predicted_filament_ids, gt_filament_ids):
    """
    Matches predicted filaments to ground truth filaments based on maximum overlap.
    Counts the number of skeletonized pixels in the ground truth labels.

    Parameters:
    - confusion_matrix (np.ndarray): 2D array representing overlaps between predicted and ground truth filaments.
    - predicted_labels (np.ndarray): 2D array of predicted filament labels.
    - ground_truth_labels (np.ndarray): 2D array of ground truth filament labels.
    - predicted_filament_ids (np.ndarray or list): Array of unique predicted filament IDs.
    - gt_filament_ids (np.ndarray or list): Array of unique ground truth filament IDs.

    Returns:
    - matches (dict): Dictionary where each key is a predicted label index, and each value is a dictionary containing:
        - 'gt_idx': Ground truth label index.
        - 'overlap_pixels': Number of overlapping pixels between the matched predicted and ground truth labels.
        - 'predicted_pixels': Total number of pixels in the predicted label.
        - 'gt_pixels': Number of skeletonized pixels in the ground truth label.
    """
    # Convert confusion matrix to cost matrix for maximization
    # The Hungarian algorithm minimizes total cost, so negate the confusion matrix to maximize overlaps
    cost_matrix = -confusion_matrix

    # Solve the linear assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Initialize the matches dictionary
    matches = {}

    # Calculate the size (number of pixels) of each predicted filament
    predicted_sizes = {pred_id: np.sum(predicted_labels == pred_id) for pred_id in predicted_filament_ids}

    # Calculate the number of skeletonized pixels for each ground truth filament
    gt_skeleton_sizes = {}
    for gt_id in gt_filament_ids:
        gt_mask = (ground_truth_labels == gt_id)
        gt_skeleton = skeletonize(gt_mask)
        gt_skeleton_sizes[gt_id] = np.sum(gt_skeleton)

    # Build the matches dictionary with 'gt_pixels'
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        pred_id = predicted_filament_ids[pred_idx]
        gt_id = gt_filament_ids[gt_idx]
        overlap_pixels = confusion_matrix[pred_idx, gt_idx]
        predicted_pixels = predicted_sizes[pred_id]
        gt_pixels = gt_skeleton_sizes[gt_id]

        matches[pred_idx] = {
            'gt_idx': gt_idx,
            'overlap_pixels': overlap_pixels,
            'predicted_pixels': predicted_pixels,
            'gt_pixels': gt_pixels
        }

    return matches


def bad_good_match(matches, predicted_filament_ids, gt_filament_ids, F1_THRESHOLD=0.70, IOU_THRESHOLD=0.50):
    # Initialize a list to store badly matched filaments
    bad_matches = []


    for pred_idx, match_info in matches.items():
        gt_idx = match_info['gt_idx']
        overlap_pixels = match_info['overlap_pixels']
        predicted_pixels = match_info['predicted_pixels']
        gt_pixels = match_info['gt_pixels']

        pred_id = predicted_filament_ids[pred_idx]
        gt_id = gt_filament_ids[gt_idx]

        # Calculate precision and recall
        precision = overlap_pixels / predicted_pixels if predicted_pixels > 0 else 0
        recall = overlap_pixels / gt_pixels if gt_pixels > 0 else 0

        # Calculate F1-Score
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate IoU
        union_pixels = predicted_pixels + gt_pixels - overlap_pixels
        iou = overlap_pixels / union_pixels if union_pixels > 0 else 0

        # Check if the match is bad based on thresholds
        is_bad_match = f1_score < F1_THRESHOLD or iou < IOU_THRESHOLD
        if is_bad_match:
            bad_matches.append(pred_idx)  # Store the index of the badly matched predicted filament

        # Per-filament match diagnostics (debug-level to keep output quiet).
        log.debug(
            "Pred %s <-> GT %s | overlap=%d pred_px=%d gt_px=%d | "
            "P=%.2f R=%.2f F1=%.2f IoU=%.2f%s",
            pred_id, gt_id, overlap_pixels, predicted_pixels, gt_pixels,
            precision, recall, f1_score, iou,
            "  **bad match**" if is_bad_match else "",
        )

    return bad_matches


def calculate_metrics(predicted_labels, ground_truth_labels, matches, confusion_matrix, predicted_filament_ids, gt_filament_ids):
    """
    Calculates performance metrics based on matched filaments and the confusion matrix.
    Includes Precision, Recall, F1-Score (Dice Coefficient), and Intersection over Union (IoU).
    
    Parameters:
    - predicted_labels (np.ndarray): 2D array of predicted filament labels.
    - ground_truth_labels (np.ndarray): 2D array of ground truth filament labels.
    - matches (dict): Dictionary mapping predicted label indices to dictionaries containing match information.
    - confusion_matrix (np.ndarray): 2D array representing overlaps between predicted and ground truth filaments.
    - predicted_filament_ids (list or np.ndarray): List of unique predicted filament IDs.
    - gt_filament_ids (list or np.ndarray): List of unique ground truth filament IDs.
    
    Returns:
    - metrics_list (list): List of dictionaries containing metrics for each matched and unmatched filament.
    - overall_metrics (dict): Dictionary containing overall aggregated metrics.
    """
    metrics_list = []

    # Per-filament sizes, mirroring match_filaments(): the predicted size is the
    # pixel count in the predicted label image; the ground-truth size is the
    # SKELETONISED pixel count (comparable to the thin predicted traces).
    predicted_sizes = {pid: int(np.sum(predicted_labels == pid)) for pid in predicted_filament_ids}
    gt_sizes = {gid: int(np.sum(skeletonize(ground_truth_labels == gid))) for gid in gt_filament_ids}

    # Two FP/FN conventions are reported side by side:
    #   * strict (primary): a predicted pixel NOT in the matched GT filament is a
    #     false positive (fp = predicted_size - tp); a GT pixel not covered by the
    #     prediction is a false negative (fn = gt_size - tp). Standard per-filament
    #     Dice/IoU, and consistent with match_filaments()/bad_good_match().
    #   * overlap (lenient): fp/fn count only pixels overlapping *other* filaments
    #     (confusion-matrix row/column sums). This ignores predicted pixels that
    #     miss every GT filament, so it is more forgiving and reads HIGHER.
    total_tp = 0
    total_fp = total_fn = 0          # strict
    total_fp_ov = total_fn_ov = 0    # overlap (lenient)

    n_pred = len(predicted_filament_ids)
    n_gt = len(gt_filament_ids)

    # Matched filaments
    for pred_idx, match_info in matches.items():
        gt_idx = match_info['gt_idx']
        tp = match_info['overlap_pixels']
        pred_id = predicted_filament_ids[pred_idx]
        gt_id = gt_filament_ids[gt_idx]

        # strict
        fp = max(0, predicted_sizes[pred_id] - tp)
        fn = max(0, gt_sizes[gt_id] - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0

        # overlap (lenient)
        fp_ov = confusion_matrix[pred_idx, :].sum() - tp
        fn_ov = confusion_matrix[:, gt_idx].sum() - tp
        precision_ov = tp / (tp + fp_ov) if (tp + fp_ov) > 0 else 0
        recall_ov = tp / (tp + fn_ov) if (tp + fn_ov) > 0 else 0
        f1_ov = (2 * precision_ov * recall_ov) / (precision_ov + recall_ov) if (precision_ov + recall_ov) > 0 else 0
        union_ov = tp + fp_ov + fn_ov
        iou_ov = tp / union_ov if union_ov > 0 else 0

        metrics_list.append({
            'predicted_label': pred_id,
            'ground_truth_label': gt_id,
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            # overlap-only (lenient) variant
            'false_positive_overlap': fp_ov,
            'false_negative_overlap': fn_ov,
            'precision_overlap': precision_ov,
            'recall_overlap': recall_ov,
            'f1_score_overlap': f1_ov,
            'iou_overlap': iou_ov,
        })
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_fp_ov += fp_ov
        total_fn_ov += fn_ov

    # Unmatched predicted filaments → pure false positives
    unmatched_predicted = set(range(n_pred)) - set(matches.keys())
    for pred_idx in unmatched_predicted:
        pred_id = predicted_filament_ids[pred_idx]
        fp = predicted_sizes[pred_id]                 # strict: the whole filament
        fp_ov = confusion_matrix[pred_idx, :].sum()   # overlap: only overlapping px
        metrics_list.append({
            'predicted_label': pred_id, 'ground_truth_label': None,
            'true_positive': 0, 'false_positive': fp, 'false_negative': 0,
            'precision': 0, 'recall': 0, 'f1_score': 0, 'iou': 0,
            'false_positive_overlap': fp_ov, 'false_negative_overlap': 0,
            'precision_overlap': 0, 'recall_overlap': 0, 'f1_score_overlap': 0, 'iou_overlap': 0,
        })
        total_fp += fp
        total_fp_ov += fp_ov

    # Unmatched ground-truth filaments → pure false negatives
    matched_gt_indices = {mi['gt_idx'] for mi in matches.values()}
    unmatched_ground_truth = set(range(n_gt)) - matched_gt_indices
    for gt_idx in unmatched_ground_truth:
        gt_id = gt_filament_ids[gt_idx]
        fn = gt_sizes[gt_id]                          # strict: the whole skeleton
        fn_ov = confusion_matrix[:, gt_idx].sum()     # overlap: only overlapping px
        metrics_list.append({
            'predicted_label': None, 'ground_truth_label': gt_id,
            'true_positive': 0, 'false_positive': 0, 'false_negative': fn,
            'precision': 0, 'recall': 0, 'f1_score': 0, 'iou': 0,
            'false_positive_overlap': 0, 'false_negative_overlap': fn_ov,
            'precision_overlap': 0, 'recall_overlap': 0, 'f1_score_overlap': 0, 'iou_overlap': 0,
        })
        total_fn += fn
        total_fn_ov += fn_ov

    def _agg(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        j = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        return p, r, f, j

    p, r, f, j = _agg(total_tp, total_fp, total_fn)
    p_ov, r_ov, f_ov, j_ov = _agg(total_tp, total_fp_ov, total_fn_ov)

    overall_metrics = {
        # strict (primary): standard per-filament Dice/IoU
        'total_true_positive': total_tp,
        'total_false_positive': total_fp,
        'total_false_negative': total_fn,
        'overall_precision': p,
        'overall_recall': r,
        'overall_f1_score': f,
        'overall_iou': j,
        # overlap-only (lenient): counts only overlapping pixels as fp/fn, so it
        # ignores predicted pixels that miss every GT filament and reads higher.
        'total_false_positive_overlap': total_fp_ov,
        'total_false_negative_overlap': total_fn_ov,
        'overall_precision_overlap': p_ov,
        'overall_recall_overlap': r_ov,
        'overall_f1_score_overlap': f_ov,
        'overall_iou_overlap': j_ov,
    }

    return metrics_list, overall_metrics


def visualize_labels(predicted_labels, ground_truth_labels) -> plt.Figure:
    """Return a figure with predicted and ground-truth labels side by side."""
    num_labels = max(predicted_labels.max(), ground_truth_labels.max()) + 1
    colors = np.random.rand(num_labels, 3)
    colors[0] = [0, 0, 0]
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('Predicted Filaments')
    axes[0].imshow(predicted_labels, cmap=cmap)
    axes[0].axis('off')
    axes[1].set_title('Ground Truth Filaments')
    axes[1].imshow(ground_truth_labels, cmap=cmap)
    axes[1].axis('off')
    fig.tight_layout()
    return fig


def visualize_overlaps(predicted_labels, ground_truth_labels) -> plt.Figure:
    """Return a figure colour-coding TP / FP / FN pixel-wise overlaps."""
    overlap = np.zeros_like(predicted_labels, dtype=int)
    overlap[(predicted_labels > 0) & (ground_truth_labels > 0)] = 2  # TP
    overlap[(predicted_labels > 0) & (ground_truth_labels == 0)] = 1  # FP
    overlap[(predicted_labels == 0) & (ground_truth_labels > 0)] = 3  # FN

    cmap = ListedColormap([
        (0, 0, 0),  # Background
        (1, 0, 0),  # FP — red
        (0, 1, 0),  # TP — green
        (0, 0, 1),  # FN — blue
    ])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Overlap Visualization  (green=TP  red=FP  blue=FN)')
    ax.imshow(overlap, cmap=cmap)
    ax.axis('off')
    fig.tight_layout()
    return fig


def visualize_matched_filaments(predicted_labels, ground_truth_labels, matches, predicted_filament_ids, gt_filament_ids, bad_matches) -> plt.Figure:
    """
    Visualizes matched filaments by assigning the same color to matched filaments.
    Badly matched filaments are highlighted differently.
    
    Parameters:
    - predicted_labels (np.ndarray): 2D array of predicted filament labels.
    - ground_truth_labels (np.ndarray): 2D array of ground truth filament labels.
    - matches (dict): Dictionary mapping predicted label indices to dictionaries containing match information.
    - predicted_filament_ids (list or np.ndarray): List of unique predicted filament IDs.
    - gt_filament_ids (list or np.ndarray): List of unique ground truth filament IDs.
    - bad_matches (list): List of predicted filament indices that are badly matched.
    """
    matched_predicted = np.zeros_like(predicted_labels, dtype=int)
    matched_ground_truth = np.zeros_like(ground_truth_labels, dtype=int)

    for pred_idx, match_info in matches.items():
        gt_idx = match_info['gt_idx']
        pred_label = predicted_filament_ids[pred_idx]
        gt_label = gt_filament_ids[gt_idx]
        # Assign a new label for visualization
        new_label = pred_idx + 1  # Start from 1 to avoid background

        # Assign a negative label for badly matched filaments
        if pred_idx in bad_matches:
            new_label = -new_label  # Negative value to indicate bad match

        matched_predicted[predicted_labels == pred_label] = new_label
        matched_ground_truth[ground_truth_labels == gt_label] = new_label

    # Handle unmatched filaments (set to background)
    unmatched_predicted = set(range(len(predicted_filament_ids))) - set(matches.keys())
    for pred_idx in unmatched_predicted:
        pred_label = predicted_filament_ids[pred_idx]
        matched_predicted[predicted_labels == pred_label] = 0  # Set to background

    # Handle unmatched ground truth filaments (set to background)
    matched_gt_indices = {match_info['gt_idx'] for match_info in matches.values()}
    unmatched_ground_truth = set(range(len(gt_filament_ids))) - matched_gt_indices
    for gt_idx in unmatched_ground_truth:
        gt_label = gt_filament_ids[gt_idx]
        matched_ground_truth[ground_truth_labels == gt_label] = 0  # Set to background

    # Create color maps
    num_labels = max(abs(matched_predicted.max()), abs(matched_ground_truth.max())) + 1
    colors = np.random.rand(num_labels, 3)
    colors[0] = [0, 0, 0]  # Background color
    # Set a specific color for badly matched filaments (e.g., red)
    bad_match_color = [1, 0, 0]  # Red color for bad matches

    # Adjust colors for bad matches
    cmap_colors = []
    for idx in range(num_labels):
        if idx == 0:
            cmap_colors.append(colors[0])
        elif -idx in matched_predicted or -idx in matched_ground_truth:
            cmap_colors.append(bad_match_color)
        else:
            cmap_colors.append(colors[idx])
    cmap = ListedColormap(cmap_colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('Matched Predicted Filaments')
    axes[0].imshow(matched_predicted, cmap=cmap)
    axes[0].axis('off')
    axes[1].set_title('Matched Ground Truth Filaments')
    axes[1].imshow(matched_ground_truth, cmap=cmap)
    axes[1].axis('off')
    fig.tight_layout()
    return fig


def visualize_false_positives_negatives(predicted_labels, ground_truth_labels, matches, predicted_filament_ids, gt_filament_ids) -> plt.Figure:
    """
    Visualizes false positives and false negatives.

    Parameters:
    - predicted_labels (np.ndarray): 2D array of predicted filament labels.
    - ground_truth_labels (np.ndarray): 2D array of ground truth filament labels.
    - matches (dict): Dictionary mapping predicted label indices to dictionaries containing match information.
    - predicted_filament_ids (list or np.ndarray): List of unique predicted filament IDs.
    - gt_filament_ids (list or np.ndarray): List of unique ground truth filament IDs.
    """
    false_positive_mask = np.zeros_like(predicted_labels, dtype=int)
    false_negative_mask = np.zeros_like(ground_truth_labels, dtype=int)

    # False Positives: Predicted filaments not matched
    unmatched_predicted = set(range(len(predicted_filament_ids))) - set(matches.keys())
    for pred_idx in unmatched_predicted:
        pred_label = predicted_filament_ids[pred_idx]
        false_positive_mask[predicted_labels == pred_label] = 1

    # False Negatives: Ground truth filaments not matched
    # Extract all matched ground truth indices
    matched_gt_indices = {match_info['gt_idx'] for match_info in matches.values()}
    unmatched_ground_truth = set(range(len(gt_filament_ids))) - matched_gt_indices
    for gt_idx in unmatched_ground_truth:
        gt_label = gt_filament_ids[gt_idx]
        false_negative_mask[ground_truth_labels == gt_label] = 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('False Positive Filaments')
    axes[0].imshow(false_positive_mask, cmap='hot')
    axes[0].axis('off')
    axes[1].set_title('False Negative Filaments')
    axes[1].imshow(false_negative_mask, cmap='hot')
    axes[1].axis('off')
    fig.tight_layout()
    return fig


def visualize_bad_matches(predicted_labels, ground_truth_labels, matches, predicted_filament_ids, gt_filament_ids, bad_matches) -> plt.Figure:
    """
    Visualizes only the badly matched filaments.
    
    Parameters:
    - predicted_labels (np.ndarray): 2D array of predicted filament labels.
    - ground_truth_labels (np.ndarray): 2D array of ground truth filament labels.
    - matches (dict): Dictionary mapping predicted label indices to dictionaries containing match information.
    - predicted_filament_ids (list or np.ndarray): List of unique predicted filament IDs.
    - gt_filament_ids (list or np.ndarray): List of unique ground truth filament IDs.
    - bad_matches (list): List of predicted filament indices that are badly matched.
    """
    bad_predicted = np.zeros_like(predicted_labels, dtype=int)
    bad_ground_truth = np.zeros_like(ground_truth_labels, dtype=int)

    for pred_idx in bad_matches:
        match_info = matches[pred_idx]
        gt_idx = match_info['gt_idx']
        pred_label = predicted_filament_ids[pred_idx]
        gt_label = gt_filament_ids[gt_idx]
        # Assign a label for visualization
        new_label = pred_idx + 1  # Start from 1 to avoid background

        bad_predicted[predicted_labels == pred_label] = new_label
        bad_ground_truth[ground_truth_labels == gt_label] = new_label

    # Create color maps
    num_labels = bad_predicted.max() + 1
    colors = np.random.rand(num_labels, 3)
    colors[0] = [0, 0, 0]  # Background color
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('Badly Matched Predicted Filaments')
    axes[0].imshow(bad_predicted, cmap=cmap)
    axes[0].axis('off')
    axes[1].set_title('Corresponding Ground Truth Filaments')
    axes[1].imshow(bad_ground_truth, cmap=cmap)
    axes[1].axis('off')
    fig.tight_layout()
    return fig


def calculate_metrics_with_mcc1(confusion_matrix, matches, predicted_filament_ids, gt_filament_ids, noise_threshold=10):
    """
    Calculates performance metrics including MCC based on the confusion matrix and filament matches.

    Parameters:
    - confusion_matrix (np.ndarray): 2D array representing overlaps between predicted and ground truth filaments.
    - matches (dict): Mapping from predicted filament indices to ground truth filament indices.
    - predicted_filament_ids (np.ndarray): Array of unique predicted filament IDs.
    - gt_filament_ids (np.ndarray): Array of unique ground truth filament IDs.
    - noise_threshold (int): Minimum number of overlapping pixels to consider significant.

    Returns:
    - metrics (dict): Dictionary containing TP, FP, FN, TN, Precision, Recall, F1-Score, and MCC.
    """
    n_pred = len(predicted_filament_ids)
    n_gt = len(gt_filament_ids)

    # Initialize counters
    tp = 0
    fp = 0
    fn = 0

    # Binarize the confusion matrix based on the noise threshold
    binary_confusion = (confusion_matrix >= noise_threshold).astype(int)

    # Initialize sets to track matched filaments
    matched_pred_indices = set()
    matched_gt_indices = set()

    # True Positives and False Positives
    for pred_idx in range(n_pred):
        # Check if the predicted filament matches any ground truth filament
        significant_gt_indices = np.where(binary_confusion[pred_idx, :] > 0)[0]
        num_significant_gt = len(significant_gt_indices)

        if num_significant_gt == 0:
            # No significant overlap; count as False Positive
            fp += 1
        else:
            # Predicted filament overlaps with one or more ground truth filaments
            # Determine if any of the ground truth filaments have not been matched yet
            unmatched_gt_indices = [gt_idx for gt_idx in significant_gt_indices if gt_idx not in matched_gt_indices]

            if len(unmatched_gt_indices) > 0:
                # Match the predicted filament with one of the unmatched ground truth filaments
                gt_idx = unmatched_gt_indices[0]
                tp += 1
                matched_pred_indices.add(pred_idx)
                matched_gt_indices.add(gt_idx)
            else:
                # All overlapping ground truth filaments have been matched; count as False Positive
                fp += 1

    # False Negatives
    for gt_idx in range(n_gt):
        if gt_idx not in matched_gt_indices:
            # Ground truth filament has not been matched; count as False Negative
            fn += 1

    # True Negatives
    # Total possible pairs minus the ones that are TP, FP, FN
    total_pairs = n_pred * n_gt
    tn = total_pairs - tp - fp - fn

    # Calculate precision, recall, F1-score, and MCC
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (
        precision + recall) if (precision + recall) > 0 else 0

    # Calculate MCC
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator > 0 else 0

    # Over-segmentation: how many predicted fragments overlap each GT filament.
    # The presence-based MCC above cannot see this — a GT filament split into
    # several predicted pieces still counts as "detected". Here we measure it
    # directly from the binarized confusion matrix.
    #   preds_per_gt[g]        = number of predicted filaments overlapping GT g
    #   oversegmentation_ratio = matched predicted fragments / GT filaments hit
    #                            (1.0 = clean 1:1; >1.0 = over-segmented)
    #   gt_oversegmented       = number of GT filaments split into >1 fragment
    preds_per_gt = binary_confusion.sum(axis=0)
    overlapped_gt = int((preds_per_gt > 0).sum())
    n_overlapping_preds = int((binary_confusion.sum(axis=1) > 0).sum())
    oversegmentation_ratio = (n_overlapping_preds / overlapped_gt) if overlapped_gt > 0 else 0.0
    gt_oversegmented = int((preds_per_gt > 1).sum())

    metrics = {
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'true_negative': tn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mcc': mcc,
        'oversegmentation_ratio': oversegmentation_ratio,
        'gt_oversegmented': gt_oversegmented
    }

    return metrics
