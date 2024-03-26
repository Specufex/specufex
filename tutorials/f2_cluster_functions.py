## p2_cluster_functions.py

## For cluster processing 
## from p2_cluster_functions import makeWfTemplate, ClusterExpand, ClusterConverge

## For Hierarchical clustering
## from p2_cluster_functions import linearizeFP, 
## from sklearn.cluster import AgglomerativeClustering


import h5py
import pandas as pd
import numpy as np
from obspy import read
from haversine import haversine
from scipy.signal import butter, lfilter
from obspy.signal.cross_correlation import correlate, xcorr_max
import datetime


from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import sklearn.metrics


try:
    from scipy.fft import fft, fftfreq
except:
    from scipy.fftpack import fft, fftfreq


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def getWF(evID,dataH5_path):
    """
    Load waveform data from H5 file and zero mean

    Parameters
    ----------
    evID : int
    .
    dataH5_path : str
    .
    Returns
    -------
    wf_zeromean : numpy array

    """

    with h5py.File(dataH5_path,'a') as fileLoad:

        wf_data_full = fileLoad[f'waveforms'].get(str(evID))[:]

    wf_data = wf_data_full

    # wf_filter = butter_bandpass_filter(wf_data, fmin,fmax,fs,order=4)
    wf_zeromean = wf_data - np.mean(wf_data)

    return wf_zeromean



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def getSgram(evID,SpecUFEx_H5_path):


    with h5py.File(SpecUFEx_H5_path,'r') as fileLoad:

        sgram = fileLoad['spectrograms'].get(str(evID))[:]
        fSTFT = fileLoad['fSTFT'].get('fSTFT')[()]
        tSTFT = fileLoad['tSTFT'].get('tSTFT')[()]
       
    
    return tSTFT, fSTFT, sgram

    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def getFP(evID,SpecUFEx_H5_path):

    with h5py.File(SpecUFEx_H5_path,'r') as MLout:

        fp = MLout['fingerprints'].get(str(evID))[:]

        return fp




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


    
def calcFFT(wf_data,lenData,fs,roll=100):
    '''


    Parameters
    ----------
    wf_data : np array
    lenData : number of samples
    fs : samploing rate
    roll : samples to calculate rolling average The default is 100.

    Returns
    -------
    rollingf : rolling average of frequency bins
    rollingFFT : rolling average of FFT

    '''

    # x = np.linspace(0.0, fs*lenData, lenData, endpoint=False)
    y = wf_data
    yf = fft(y)
    xf = fftfreq(lenData, 1/fs)[:lenData//2]
    real_fft = 2.0/lenData * np.abs(yf[0:lenData//2])

    df_spectra = pd.DataFrame({'fft':real_fft,
                               'f':xf})

    rollingFFT = df_spectra.fft.rolling(roll, min_periods=1).mean()
    rollingf = df_spectra.f.rolling(roll, min_periods=1).mean()



    return rollingf, rollingFFT

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################






    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def linearizeFP(SpecUFEx_H5_path,ev_IDs):
    """
    Linearize fingerprints, stack into array 

    Parameters
    ----------
    SpecUFEx_H5_path :str
    cat00 : pandas dataframe

    Returns
    -------
    X : numpy array
        (Nevents,Ndim)

    """

    X = []
    with h5py.File(SpecUFEx_H5_path,'r') as MLout:
        for evID in ev_IDs:
            fp = MLout['fingerprints'].get(str(evID))[:]
            linFP = fp.reshape(1,len(fp)**2)[:][0]
            X.append(linFP)

    X = np.array(X)

    return X





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def PVEofPCA(X,numPCMax=100,cum_pve_thresh=.8,stand='MinMax'):
    """
    Parameters
    ----------
    X : numpy array
        Linearized fingerprints.
    numPCMax : int, optional
        Maximum number of principal components. The default is 100.
    cum_pve_thresh : int or float, optional
        Keep PCs until cumulative PVE reaches threshold. The default is .8.
    stand : str, optional
        Parameter for SKLearn's StandardScalar(). The default is 'MinMax'.

    Returns
    -------
    PCA_df : pandas dataframe
        Columns are PCs, rows are event.
    numPCA : int
        Number of PCs calculated.
    cum_pve : float
        Cumulative PVE.

=======

    Returns
    -------
    PCA_df : pandas dataframe
        Columns are PCs, rows are event.
    numPCA : int
        Number of PCs calculated.
    cum_pve : float
        Cumulative PVE.


    """

    if stand=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif stand=='MinMax':
        X_st = MinMaxScaler().fit_transform(X)
    else:
        X_st = X


    numPCA_range = range(1,numPCMax)


    for numPCA in numPCA_range:

        sklearn_pca = PCA(n_components=numPCA)

        Y_pca = sklearn_pca.fit_transform(X_st)

        pve = sklearn_pca.explained_variance_ratio_

        cum_pve = pve.sum()
        print(numPCA,cum_pve)
        if cum_pve >= cum_pve_thresh:

            print('break')
            break



    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]

    PCA_df = pd.DataFrame(data = Y_pca, columns = pc_cols)


    return PCA_df, numPCA, cum_pve




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def calcSilhScore(X,range_n_clusters):
    """
    Calculates the optimal number of clusters and silhouette scores for K-means clustering.

    Parameters:
    -----------
    X : array-like
        The input dataset to be clustered.
    range_n_clusters : list or range
        A range of integers specifying the number of clusters to evaluate.

    Returns:
    --------

    ## Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters
    ## Returns altered cat00 dataframe with cluster labels and SS scores,
    ## Returns NEW catall dataframe with highest SS scores

    Kopt : int
        The optimal number of clusters.
    maxSilScore : float
        The maximum silhouette score achieved.
    avgSils : list
        A list of average silhouette scores for each number of clusters.
    sse : list
        A list of sum of squared errors for each number of clusters.
    cluster_labels_best : array-like
        Cluster labels for the best clustering result.
    ss_best : array-like
        Silhouette scores for each sample in the best clustering result.
    euc_dist_best : array-like
        Euclidean distances to the centroids for each sample in the best clustering result.
    """





## alt. X = 'PCA'

    maxSilScore = 0

    sse = []
    avgSils = []
    centers = []

    for n_clusters in range_n_clusters:

        print(f"kmeans on {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters,
                           max_iter = 500,
                           init='k-means++', #how to choose init. centroid
                           n_init=10, #number of Kmeans runs
                           random_state=0) #set rand state

        #get cluster labels
        cluster_labels_0 = kmeans.fit_predict(X)

        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]

        #get euclid dist to centroid for each point
        sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
        sum_sqr_dist = sqr_dist.sum(axis=1)
        euc_dist = np.sqrt(sum_sqr_dist)

        #save centroids
        centers.append(kmeans.cluster_centers_ )

        #kmeans loss function
        sse.append(kmeans.inertia_)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

#         %  Silhouette avg
        avgSil = np.mean(sample_silhouette_values)

        # avgSil = np.median(sample_silhouette_values)

        avgSils.append(avgSil)
        if avgSil > maxSilScore:
            Kopt = n_clusters
            maxSilScore = avgSil
            cluster_labels_best = cluster_labels
            euc_dist_best = euc_dist
            ss_best       = sample_silhouette_values


    print(f"Best cluster: {Kopt}")



    return Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def calcSilhScoreHierarch(X,range_n_clusters):
    """

    Parameters
    ----------

    Returns
    -------

    """

## Return avg silh scores from GMM, avg SSEs, and Kopt for 2:Kmax clusters
## Returns altered cat00 dataframe with cluster labels and SS scores,
## Returns NEW catall dataframe with highest SS scores


## alt. X = 'PCA'

    maxSilScore = 0

    sse = []
    avgSils = []
    centers = []

    for n_clusters in range_n_clusters:

        print(f"hierarchical clustering with {n_clusters} clusters...")


        hierarch = AgglomerativeClustering(distance_threshold=None, 
                                    n_clusters=n_clusters,
                                    linkage='ward')


        #get cluster labels
        cluster_labels_0 = hierarch.fit_predict(X)

        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]

        #save centroids
#         centers.append(gmm.means_ )


        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

#         %  Silhouette avg
        avgSil = np.mean(sample_silhouette_values)

        # avgSil = np.median(sample_silhouette_values)

        avgSils.append(avgSil)
        if avgSil > maxSilScore:
            Kopt = n_clusters
            maxSilScore = avgSil
            cluster_labels_best = cluster_labels
            ss_best = sample_silhouette_values


    print(f"Best cluster: {Kopt}")



    return Kopt, maxSilScore, avgSils, cluster_labels_best, ss_best


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def getTopFCat(cat,topF=1,startInd=0,distMeasure = "SilhScore"):
    """
    Make dataframe of most representative events


    Parameters
    ----------
    cat : pandas.DataFrame
        Catalog of events, must have 'Cluster', 'SS', and/or 'euc_dist' columns
    topf : int, default 1
        Number of top 'F' events in each cluster
    startInd : int, default 0
        Index to start counting top events
    distMeasure : {'SilhScore','EucDist'} default 'SilhScore'
        Type of clustering measure

    Returns
    -------
    cat_topF : pandas.DataFrame
        top {topF} of each cluster in catalog

    """

    cat_topF = pd.DataFrame();


    Kopt = np.max(cat.Cluster.unique())

    for k in range(1,Kopt+1):

        cat0 = cat.where(cat.Cluster==k).dropna();

        if distMeasure == "SilhScore":
            cat0 = cat0.sort_values(by='SS',ascending=False)

        if distMeasure == "EucDist":
            cat0 = cat0.sort_values(by='euc_dist',ascending=True)

        try:
            cat0 = cat0[startInd:startInd+topF]
        except: #if less than topF number events in cluster
            print(f"{startInd+topF} is larger than cluster {k} size (n={len(cat0)})")
            cat0 = cat0




        cat_topF = cat_topF.append(cat0);


    ## sometimes these types get changed in the process?
    ## can comment out next two lines
    cat_topF['Cluster'] = [int(c) for c in cat_topF.Cluster];

    cat_topF['datetime_index'] = [pd.to_datetime(d) for d in cat_topF.datetime];

    if distMeasure == "SilhScore":
        cat_topF = cat_topF.sort_values(by =['Cluster','SS'])


    if distMeasure == "EucDist":
        cat_topF = cat_topF.sort_values(by =['Cluster','euc_dist'])


    cat_topF = cat_topF.set_index('datetime_index')


    return cat_topF




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def getTopFCatProp(cat,topP=.1,distMeasure = "SilhScore"):
    """
    Make dataframe of most representative events


    Parameters
    ----------
    cat : pandas.DataFrame
        Catalog of events, must have 'Cluster', 'SS', and/or 'euc_dist' columns
    topf : int, default 1
        Number of top 'F' events in each cluster
    startInd : int, default 0
        Index to start counting top events
    distMeasure : {'SilhScore','EucDist'} default 'SilhScore'
        Type of clustering measure

    Returns
    -------
    cat_topF : pandas.DataFrame
        top {topF} of each cluster in catalog

    """

    cat_topF = pd.DataFrame();


    Kopt = np.max(cat.Cluster.unique())

    for k in range(1,Kopt+1):

        cat0 = cat.where(cat.Cluster==k).dropna();

        P = np.quantile(a=cat0.SS,q=1-topP)
        
        cat0 = cat0[cat0.SS>=P]

        cat_topF = cat_topF.append(cat0);


    ## sometimes these types get changed in the process?
    ## can comment out next two lines
    cat_topF['Cluster'] = [int(c) for c in cat_topF.Cluster];

    cat_topF['datetime_index'] = [pd.to_datetime(d) for d in cat_topF.datetime];

    if distMeasure == "SilhScore":
        cat_topF = cat_topF.sort_values(by =['Cluster','SS'])


    if distMeasure == "EucDist":
        cat_topF = cat_topF.sort_values(by =['Cluster','euc_dist'],ascending=True)


    cat_topF = cat_topF.set_index('datetime_index')


    return cat_topF




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


