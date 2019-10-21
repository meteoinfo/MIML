"""Hierarchical Agglomerative Clustering

These routines perform some hierarchical agglomerative clustering of some
input data.

Authors: Yaqiang Wang
License: LGPL
"""
from smile.clustering import HierarchicalClustering as JHierarchicalClustering
from smile.clustering.linkage import CompleteLinkage, SingleLinkage, UPGMALinkage, UPGMCLinkage, \
    WardLinkage, WPGMALinkage, WPGMCLinkage
import mipylib.numeric as np

from .cluster import Cluster

class HierarchicalClustering(Cluster):
    '''
    Agglomerative Hierarchical Clustering.
    
    Hierarchical agglomerative clustering seeks to build a hierarchy of clusters in a bottom up 
    approach: each observation starts in its own cluster, and pairs of clusters are merged as one 
    moves up the hierarchy. The results of hierarchical clustering are usually presented in a 
    dendrogram.
    
    :param proximity: (*array*) The proximity matrix to store the distance measure of
        dissimilarity. To save space, we only need the lower half of matrix.
    :param method: (*string*) The agglomeration method to merge clusters. This should be one of
        "single", "complete", "upgma", "upgmc", "wpgma", "wpgmc", and "ward".
    '''
    
    def __init__(self, proximity, method='single'):
        self._proximity = proximity.tojarray('double')
        self._linkage = self._get_linkage(method)
        self._model = JHierarchicalClustering(self._linkage)
        
    def _get_linkage(self, method):
        if method == "single":
            return SingleLinkage(self._proximity)
        if method == "complete":
            return CompleteLinkage(self._proximity)
        if method == "upgma" or method == "average":
            return UPGMALinkage(self._proximity)
        if method == "upgmc" or method == "centroid":
            return UPGMCLinkage(self._proximity)
        if method == "wpgma":
            return WPGMALinkage(self._proximity)
        if method == "wpgmc" or method == "median":
            return WPGMCLinkage(self._proximity)
        if method == "ward":
            return WardLinkage(self._proximity)
        else:
            return None
            
    def partition(self, k):
        '''
        Cuts a tree into several groups by specifying the desired number.
        
        :param k: (*int*) The number of clusters.
        
        :returns: (*array*) The cluster label of each sample.
        '''
        r = self._model.partition(k)
        return np.array(r)
        
        
##############################################