
from cython.parallel import prange
import numpy as np
cimport numpy as np
from cython cimport view, boundscheck, wraparound
from libc.math cimport fabs, log, exp, INFINITY
#from numpy.math cimport logl, expl, INFINITY

@boundscheck(False)
@wraparound(False)
def update_neighs(double[:,:] mu,
              double[:,:] docneig_signal,
              double[:,:] wordneig_signal,
              int[:] w, int[:] d, 
              double[:] x, const int K, const int N):
    """ Compute row- and column-wise summary statistics
    
    Given mu, docneig_signal and wordneig_signal is determined
    which constitute the column-wise and row-wise sum of the cluster
    assignments. This function is used to initialize these arrays
    before belief propagation is utilized.
    
    Parameters
    ----------
    mu : np.array[N, K]
        Soft-cluster assignments for all words
    docneig_signal : np.array[V, K]
        Sum of cluster assignments for each word across documents.
    wordneig_signal : np.array[D, K]
        Sum of cluster assignments for each document across words
    w : np.array[N]
        Word indices
    d :  np.array[N]
        Document indices
    x : np.array[N]
        Word count
    K : int
        Number of topics
    N : int
        Total number of words across all documents. Corresponds to len(x).
    """
    
    cdef int i, k
    cdef double v
    
    for i in prange(N, nogil=True):
        for k in range(K):
            v = x[i]*mu[i,k]
            docneig_signal[w[i], k] += v
            wordneig_signal[d[i], k] += v


@boundscheck(False)
@wraparound(False)
def update_mu_async(double[:,:] mu,
              double[:,:] docneig_signal,
              double[:,:] wordneig_signal,
              int[:] w, int[:] d, 
              double[:] x, double alpha, 
              double beta, const int K, const int N):
    """
    Updates the topic assignments for all words using belief propagation.
    
    Using the current topic assignments mu across words in the document
    and across documents for a given word, it updates the mu's across all words
    
    Parameters:
    mu : np.array[N, K]
        Soft-cluster assignments for all words
    docneig_signal : np.array[V, K]
        Sum of cluster assignments for each word across documents.
    wordneig_signal : np.array[D, K]
        Sum of cluster assignments for each document across words
    w : np.array[N]
        Word indices
    d :  np.array[N]
        Document indices
    x : np.array[N]
        Word count
    alpha : float
        Document-topic prior.
    beta : float
        Word-topic prior.
    K : int
        Number of topics
    N : int
        Total number of words across all documents. Corresponds to len(x).    

    Returns :
      float: mean absolute change of the topic assignments mu.
    """
    
    cdef int i, w_i, d_i, k, n_words
    cdef double[::view.contiguous] mu_mw = np.zeros([K], dtype=np.float)
    cdef double[::view.contiguous] mu_md = np.zeros([K], dtype=np.float)
    cdef double[::view.contiguous] mu_new = np.zeros([K], dtype=np.float)
    cdef double[::view.contiguous] Nw = np.zeros(K, dtype=np.float)
    cdef double musum = 0.0
    cdef double diff = 0.0

    n_words = docneig_signal.shape[0]
    
    # init Nw
    for i in range(wordneig_signal.shape[0]):
        for k in range(K):
            Nw[k] += wordneig_signal[i, k]
    
    # compute new mu
    for i in prange(N, nogil=True):
        w_i = w[i]
        d_i = d[i]

        musum = 0.0
        for k in range(K):
            mu_md[k] = wordneig_signal[d_i, k] + beta - x[i]*mu[i, k]
            musum += mu_md[k]
        
        for k in range(K):
            mu_md[k] /= musum

        musum = 0.0
        for k in range(K):
            mu_mw[k] = (docneig_signal[w_i, k] + alpha - \
                        x[i]*mu[i, k]) / (Nw[k] - \
                        wordneig_signal[d_i, k] + alpha * n_words)
            mu_new[k] = mu_mw[k] * mu_md[k]
            musum += mu_new[k]
        
        for k in range(K):
            mu_new[k] /= musum
            
            diff += fabs(mu_new[k]- mu[i, k])
            # update the summary statistics with the new mu assignment
            # and update mu.
            docneig_signal[w_i, k] += x[i]*(mu_new[k] - mu[i, k])
            wordneig_signal[d_i, k] += x[i]*(mu_new[k] - mu[i, k])
            Nw[k] += x[i]*(mu_new[k] - mu[i, k])
            
            mu[i, k] = mu_new[k]
            
    return diff / N


@boundscheck(False)
@wraparound(False)
def _loglikelihood(int[:] w, int[:] d, 
              double[:] x,
              double[:,:] doc_topic,
              double[:,:] word_topic):
    """
    Computes the log-likelihood for the current model
    
    Parameters:
    doc_topic : np.array[V, K]
        Document-topic probabilities
    wordneig_signal : np.array[D, K]
        Word-topic probabilities
    w : np.array[N]
        Word indices
    d :  np.array[N]
        Document indices
    x : np.array[N]
        Word count

    Returns :
      float: log-likelihood
    """
    
    cdef int i, w_i, d_i, k, N, n_topics = doc_topic.shape[1]
    cdef double[::view.contiguous] buff = np.zeros(n_topics, dtype=np.float)
    cdef double loglikeli = 0.0
    cdef double maxlog = 0.0
    cdef double likeli = 0.0

    N = w.shape[0]
    
    # compute new mu
    for i in prange(N, nogil=True):
        w_i = w[i]
        d_i = d[i]

        maxlog = -INFINITY
        for k in range(n_topics):
            buff[k] = log(doc_topic[d_i, k]) + log(word_topic[w_i, k])
            if maxlog < buff[k]:
                maxlog = buff[k]

        likeli = 0.0
        for k in range(n_topics):
            likeli += exp(buff[k] - maxlog)

        loglikeli += log(likeli) + maxlog

    return loglikeli
