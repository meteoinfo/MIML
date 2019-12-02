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
from ..utils import smile_util

class HierarchicalClustering(Cluster):
    '''
    Agglomerative Hierarchical Clustering.
    
    Hierarchical agglomerative clustering seeks to build a hierarchy of clusters in a bottom up 
    approach: each observation starts in its own cluster, and pairs of clusters are merged as one 
    moves up the hierarchy. The results of hierarchical clustering are usually presented in a 
    dendrogram.
    
    :param proximity: (*array*) The proximity matrix to store the distance measure of
        dissimilarity. To save space, we only need the lower half of matrix.
    :param k: (*int*) The cluster number.
    :param linkage: (*string*) The agglomeration linkage to merge clusters. This should be one of
        "single", "complete", "upgma", "upgmc", "wpgma", "wpgmc", and "ward".
    '''
    
    def __init__(self, k=2, linkage='single'):
        super(HierarchicalClustering, self).__init__()
        
        self._k = k
        self._linkage = linkage
        
    def _get_linkage(self, linkage, proximity):
        '''
        Get linkage.
        
        :param linkage: (*string*) Linkage string.
        :param proximity: (*array*) The proximity matrix to store the distance measure of
            dissimilarity. To save space, we only need the lower half of matrix.
            
        :returns: Linkage.
        '''
        proximity = proximity.tojarray('double')
        if linkage == "single":
            return SingleLinkage(proximity)
        if linkage == "complete":
            return CompleteLinkage(proximity)
        if linkage == "upgma" or linkage == "average":
            return UPGMALinkage(proximity)
        if linkage == "upgmc" or linkage == "centroid":
            return UPGMCLinkage(proximity)
        if linkage == "wpgma":
            return WPGMALinkage(proximity)
        if linkage == "wpgmc" or linkage == "median":
            return WPGMCLinkage(proximity)
        if linkage == "ward":
            return WardLinkage(proximity)
        else:
            return None
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        proximity = smile_util.pdist(x)
        linkage = self._get_linkage(self._linkage, proximity)
        self._model = JHierarchicalClustering(linkage)
        return self
    
    def fit_predict(self, x):
        """
        Fitting and cluster data.

        :param x: (*array*) Input data.
        
        :returns: (*array*) The cluster labels.
        """
        self.fit(x)
        
        r = self._model.partition(self._k)
        return np.array(r)
        
        
##############################################