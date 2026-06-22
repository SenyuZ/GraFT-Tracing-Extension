#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filament orientation / circular-statistics analysis  (future work — not wired in).

This module was prototyped during the project to quantify *filament orientation*
relative to the major cell axis and summarise it with circular statistics
(mean/variance of angle, a polar histogram). It is **not** part of the tracing
pipeline in ``GraFT_workflow_*.py`` and is **not** imported by ``utilsGraFT`` —
it is kept here, isolated, as a starting point for the "filament orientation"
direction noted in the README's *Limitations & future work* section.

It is therefore untested on the current data and intentionally excluded from the
core dependencies. Running it needs a few **optional** packages:

    pip install astropy plotly kaleido      # circular stats + polar plot export

Typical (future) usage, once a traced graph + cell mask are available::

    from extras.orientation_analysis import filament_info, circ_stat
    info = filament_info(img_o, graphTagg, posL, out_dir, imF, maskDraw)
    mean_angle, var = circ_stat(info, out_dir)

History: these functions previously lived at the tail of ``utilsGraFT.py`` but
were never called by ``main()``; they were moved here so the core module stays
focused on tracing and free of the heavy ``astropy`` dependency.
"""

from collections import Counter

import numpy as np
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import skimage
import astropy.stats
from astropy import units as u


def tagsU(graph):
    '''
    create list with unique filament tags

    Parameters
    ----------
    graph : nx graph
        graph with defined filament tags.

    Returns
    -------
    un_tags : list
        unique filament tags.

    '''
    film = list(graph.edges(data='filament'))
    filament_tag = [film[i][2] for i in range(len(film))]
    un_tags = np.unique(filament_tag)
    return un_tags

def pos_filament(graph,pos):
    '''
    for each unique filament tag, find the start and end nodes

    Parameters
    ----------
    graph : nx graph
        graph with defined filament tags.
    pos : list
        list of node positions.

    Returns
    -------
    posF : list
        list of start and end nodes for each unique filament tag.

    '''

    unT = tagsU(graph)
    #cost matrix
    posF = np.zeros((len(unT),2))
    #find dangling nodes
    for l in range(len(unT)):
        #find start/end nodes in filaments
        tag_val = unT[l]
        nodesF = np.asarray([(a,b) for a,b, attrs in graph.edges(data=True) if attrs["filament"] == tag_val]).flatten()
        countN = np.asarray(list(Counter(nodesF).items()))
        nodeES = [val for is_good, val in zip(countN[:,1]==1, countN[:,0]) if is_good]

        if(len(nodeES)==0):
            nodesF = np.asarray([(a,b) for a,b, attrs in graph.edges(data=True)
                                  if attrs["filament dangling"] == 1 and attrs["filament"] == tag_val]).flatten()
            countN = np.asarray(list(Counter(nodesF).items()))
            nodeES = [val for is_good, val in zip(countN[:,1]==2, countN[:,0]) if is_good]
            if(len(nodeES)==0):
                #circle, and is emtpy
                nodeES = [nodesF[0]]
            nodeS1 = nodeES[0]
            nodeE1 = nodeES[0]
        elif(len(nodeES)==1):
            nodeS1 = nodeES[0]
            nodeE1 = nodeES[0]
        else:
            nodeS1 = nodeES[0]
            nodeE1 = nodeES[1]

        posF[l] = nodeS1,nodeE1
    return posF


###############################################################################
#
# post processing
#
###############################################################################

def mask2rot(mask):
    '''
    calculation of major cell axis

    Parameters
    ----------
    mask : array
        mask of cell binary image.

    Returns
    -------
    directionVector : float
        angle for major cell axis.

    '''

    skeletonizedMask = skimage.morphology.skeletonize(mask)
    coordinatesSkeleton = np.array(np.where(skeletonizedMask > 0)).T[:, ::-1]
    pointsOnSkeleton = int(len(coordinatesSkeleton) * 0.2)
    coordinateCellAxis1 = coordinatesSkeleton[pointsOnSkeleton]
    coordinateCellAxis2 = coordinatesSkeleton[-pointsOnSkeleton]
    directionVector = coordinateCellAxis2 - coordinateCellAxis1
    return directionVector

def angle_majorCell(nodes_edge,nodePositions,vec_mask):
    '''
    angle of filament from major cell axis

    Parameters
    ----------
    nodes_edge : list
        DESCRIPTION.
    nodePositions : list
        positions of nodes.
    vec_mask : float
        angle for major cell axis.

    Returns
    -------
    angle : float
        angle for filament from major cell axis.

    '''

    vec1 = nodePositions[int(nodes_edge[1])] - nodePositions[int(nodes_edge[0])]
    vec2 = vec_mask
    if((vec1==0).all()):
        angle=0
    else:
        angle=np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) * 180.0 / np.pi
    return angle


def barplot180(list_points,list_bins,save_dir,name_save,color_code):
    '''
    creation of circular barplot for angles

    Parameters
    ----------
    list_points : list
        list of binned weighted angles.
    list_bins : list
        list of bin values.
    save_dir : directory path
        path to save image.
    name_save : name
        name of figure.
    color_code : cmap
        value for cmap.

    Returns
    -------
    None.

    '''
    fig = go.Figure(go.Barpolar(
        r=list_points,
        theta=list_bins,
        width=5,
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.8,
        marker_color=color_code,
        # yellow '#FFFF00'
        # blue 	#0000FF
    ))
    fig.show()

    fig.update_layout(
        polar = dict(radialaxis = dict(showticklabels=False, ticks=''), sector = [0,180],
                     radialaxis_showgrid=False,
                     angularaxis=dict(
                         #showgrid=False,
                #rotation=180,
                #direction='clockwise',
                tickfont = dict(size = 30))
                             )
                )

    fig.write_image(save_dir+name_save, format='png')
    return




def filament_info(img_o, graphTagg, posL, path,imF,maskDraw):
    '''
    creation of dataframe to be saved as csv of all filament info of traced filaments

    Parameters
    ----------
    img_o : array
        array of raw image timeseries.
    graphTagg : nx graph
        list of traced graph.
    posL : list
        list of list of node positions.
    path : directory path
        path to save file.
    imF : arrray
        skeletonized image.
    maskDraw : array
        binary mask of cell.

    Returns
    -------
    FullfilInfo : dataframe
        dataframe containing all the data-image information.
    '''

    FullfilInfo = pd.DataFrame()


    filamentTags = np.unique(np.asarray(list(graphTagg.edges(data='filament')))[:,2])
    filInfo = pd.DataFrame()
    filamentT = nx.to_pandas_edgelist(graphTagg)
    nodesSE = pos_filament(graphTagg,posL)
    fullL = np.zeros(len(filamentTags))
    nodeD = np.zeros(len(filamentTags))
    fullI = np.zeros(len(filamentTags))
    fullC = np.zeros(len(filamentTags))
    fullBL = np.zeros(len(filamentTags))
    fullE = np.zeros(len(filamentTags))
    fullEd = np.zeros(len(filamentTags))
    fullDe = np.ones(len(filamentTags))
    tags = np.ones(len(filamentTags))
    vec_mask = mask2rot(maskDraw)
    #in case its a perfect square
    if(all(vec_mask==[0,0])):
        vec_mask=[1,0]
    for i,l in zip(filamentTags,range(len(filamentTags))):
        #filament length
        fullL[l] = np.sum(filamentT['fdist'][filamentT['filament']==i])
        fullE[l] = np.sum(filamentT['edist'][filamentT['filament']==i])
        #node distance
        nodeD[l] = np.linalg.norm(posL[int(nodesSE[l][0])]-posL[int(nodesSE[l][1])])
        #filament intensity
        fullI[l] = np.sum(filamentT['weight'][filamentT['filament']==i])
        #filament intensity by filament length
        fullC[l] = fullI[l]/fullL[l]
        #rod length over filament length
        fullBL[l] = nodeD[l]/fullL[l]
        #angle from major cell axis
        edge_ang = angle_majorCell(nodesSE[l],posL,vec_mask)
        tags[l] = i
        fullEd[l] = edge_ang #min(edge_ang,np.abs(edge_ang-180))
        # density:
    fullDe = fullDe*np.sum(imF*1)/(np.sum(maskDraw))


    filInfo['filament'] = tags
    filInfo['filament length'] = fullL
    filInfo['filament edist'] = fullE
    filInfo['filament rod length'] = nodeD
    filInfo['filament intensity'] = fullI
    filInfo['filament intensity per length'] = fullC
    filInfo['filament bendiness'] = fullBL
    filInfo['filament angle'] = fullEd
    filInfo['filament density'] = fullDe

    FullfilInfo = filInfo

    FullfilInfo.to_csv(path+'traced_filaments_info.csv',index=False)

    return FullfilInfo



def circ_stat(pd_fil_info,path):

    data = np.asarray(pd_fil_info['filament angle'])*u.deg
    weight = pd_fil_info['filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))

    hist180,bins180 = np.histogram(0,int(180/5),[0,180])

    list_ec = np.zeros(len(bins180[1:]))
    for l in range(len(bins180[1:])):

        list_ec[l] = pd_fil_info['filament length'][(pd_fil_info['filament angle']>bins180[l]) & (pd_fil_info['filament angle']<=bins180[l+1])].sum()


    bins180 = bins180[1:]-2.5
    name180 = "circ_stat/stat.png"
    barplot180(list_ec,bins180,path,name180,color_code='#0000FF')
    return mean_angle,var_val
