
from cython.parallel import prange
import numpy as np
cimport numpy as np
from cython cimport view, boundscheck, wraparound
from libc.math cimport fabs, log, exp, INFINITY

@boundscheck(False)
@wraparound(False)
def update_neighs(double[:,:] mu,
              double[:,:] word_topic_matrix,
              double[:,:] topic_document_matrix,
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
    word_topic_matrix : np.array[V, K]
        Sum of cluster assignments for each word across documents.
    topic_document_matrix : np.array[K, D]
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
            word_topic_matrix[w[i], k] += v
            topic_document_matrix[k, d[i]] += v


@boundscheck(False)
@wraparound(False)
def update_mu_async(double[:,:] mu,
              double[:,:] word_topic_matrix,
              double[:,:] topic_document_matrix,
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
    word_topic_matrix : np.array[V, K]
        Sum of cluster assignments for each word across documents.
    topic_document_matrix : np.array[K, D]
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

    n_words = word_topic_matrix.shape[0]
    
    # init Nw
    for k in range(K):
        for i in range(topic_document_matrix.shape[1]):
            Nw[k] += topic_document_matrix[k, i]
    
    # compute new mu
    for i in prange(N, nogil=True):
        w_i = w[i]
        d_i = d[i]

        musum = 0.0
        for k in range(K):
            mu_md[k] = topic_document_matrix[k,d_i] + beta - x[i]*mu[i, k]
            musum += mu_md[k]
        
        for k in range(K):
            mu_md[k] /= musum

        musum = 0.0
        for k in range(K):
            mu_mw[k] = (word_topic_matrix[w_i, k] + alpha - \
                        x[i]*mu[i, k]) / (Nw[k] - \
                        topic_document_matrix[k, d_i] + alpha * n_words)
            mu_new[k] = mu_mw[k] * mu_md[k]
            musum += mu_new[k]
        
        for k in range(K):
            mu_new[k] /= musum
            
            diff += fabs(mu_new[k]- mu[i, k])
            # update the summary statistics with the new mu assignment
            # and update mu.
            word_topic_matrix[w_i, k] += x[i]*(mu_new[k] - mu[i, k])
            topic_document_matrix[k, d_i] += x[i]*(mu_new[k] - mu[i, k])
            Nw[k] += x[i]*(mu_new[k] - mu[i, k])
            
            mu[i, k] = mu_new[k]
            
    return diff / N


@boundscheck(False)
@wraparound(False)
def collapsed_gibbs_sampling(int[:] word,
                             int[:] doc,
                             double[:] multiple,
                             int[:] z_assign,
                             double[:,:] word_topic_matrix,
                             double[:,:] topic_document_matrix,
                             double[:] rands,
                             int num_words,
                             int num_topics,
                             int voc_size,
                             double alpha,
                             double beta,
                             double [:] doc_counts,
                             double [:] topic_counts):

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
    
    cdef int i, w_i, d_i, z_i, k, n_words, test
    cdef double prob_norm, m_i, r, p
    cdef double[::view.contiguous] prob_z = np.zeros(num_topics, dtype=np.float)

    test=0
    with nogil:
        for i in range(num_words):
            w_i = word[i]
            d_i = doc[i]
            m_i = multiple[i]
            z_i = z_assign[i]

            doc_counts[d_i] -= m_i
            topic_counts[z_i] -= m_i
            word_topic_matrix[w_i, z_i] -= m_i
            topic_document_matrix[z_i, d_i] -= m_i

            prob_norm = 0.0
            for k in range(num_topics):
                prob_z[k] = (word_topic_matrix[w_i, k] + beta)/(topic_counts[k] + voc_size*beta) * (topic_document_matrix[k, d_i] + alpha)/(doc_counts[d_i] + num_topics*alpha)
                prob_norm += prob_z[k]
                 
            p = 0.0
            for k in range(num_topics):
                p += prob_z[k]/prob_norm
                if rands[i] <= p:
                    z_i = k
                    break

            z_assign[i] = z_i
            doc_counts[d_i] += m_i
            topic_counts[z_i] += m_i
            word_topic_matrix[w_i, z_i] += m_i
            topic_document_matrix[z_i, d_i] += m_i

    return 0


@boundscheck(False)
@wraparound(False)
def _marginal_loglikelihood(double loglikeli, 
              double[:,:] wt, 
              double[:,:] td,
              int num_words,
              int num_docs,
              int num_topics):
    """
    Computes the log-likelihood for the current model
    
    Parameters:
    loglikeli : double
        pre-initizalized log-likelihood
    wt : np.array[W, T]
        Word-topic counts
    td : np.array[T, D]
        Topic-document counts
    num_words : int
        Number of words
    num_docs :  int
        Number of documents
    num_topics : int
        Number of topics

    Returns :
      float: log-likelihood
    """
    
    cdef int i, j, k
    cdef double[::view.contiguous] buff = np.zeros(num_topics, dtype=np.float)
    cdef double  maxlog, likeli

    # compute new mu
    with nogil:
        for i in range(num_words):
            for j in range(num_docs):
                
                maxlog = -INFINITY
                for k in range(num_topics):
                    buff[k] = wt[i, k] + td[k, j]
                    if maxlog < buff[k]:
                        maxlog = buff[k]

                likeli = 0.0
                for k in range(num_topics):
                    likeli += exp(buff[k] - maxlog)

                loglikeli += log(likeli) + maxlog

    return loglikeli


@boundscheck(False)
@wraparound(False)
def _loglikelihood(double loglikeli, 
              double[:,:] wt, 
              double[:,:] td,
              int num_words,
              int num_docs,
              int num_topics):
    """
    Computes the log-likelihood for the current model
    
    Parameters:
    loglikeli : double
        pre-initizalized log-likelihood
    wt : np.array[W, T]
        Word-topic counts
    td : np.array[T, D]
        Topic-document counts
    num_words : int
        Number of words
    num_docs :  int
        Number of documents
    num_topics : int
        Number of topics

    Returns :
      float: log-likelihood
    """
    
    cdef int i, j, k
    cdef double[::view.contiguous] buff = np.zeros(num_topics, dtype=np.float)
    cdef double  maxlog, likeli

    # compute new mu
    with nogil:
        for i in range(num_words):
            for j in range(num_docs):
                
                maxlog = -INFINITY
                for k in range(num_topics):
                    buff[k] = wt[i, k] + td[k, j]
                    if maxlog < buff[k]:
                        maxlog = buff[k]

                likeli = 0.0
                for k in range(num_topics):
                    likeli += exp(buff[k] - maxlog)

                loglikeli += log(likeli) + maxlog

    return loglikeli


