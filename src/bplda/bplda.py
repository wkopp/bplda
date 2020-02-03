"""Main module."""
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import coo_matrix
from sklearn.utils import check_random_state
from bplda._bplda import update_neighs
from bplda._bplda import update_mu_async
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class LDA(BaseEstimator, TransformerMixin):
    """
    Latent dirichlet allocation model with belief propagation
    based on the mean field approximation.
    The algorithm is based on
    Zeng, Cheung, Liu, Learning Topic Models by belief propagation. 2012.
    IEEE Transactions on Pattern analysis and machine learning.

    Examples
    --------

    .. code-block:: python

       import numpy as np
       from bplda import LDA

       # toy data (10 documents, vocabulary size=5)
       minitest = np.zeros((5, 10))
       minitest[:3,:5]=1
       minitest[-3:,-5:]=1

       model = LDA(3, niter=10, seed=10)
       doc_top = model.fit_transform(minitest)

       model.word_topic_


    Parameters
    -----------

    n_topics : int
        Number of topics
    alpha : float
        Word-topic prior. Default: 0.1
    beta : float
        Document-topic prior. Default: 0.1
    niter : int
        Number updates using belief propagation.
    seed : int or None
        Random seed for reproducibility
    verbose : boolean
        Prints progress per iteration.
    debug : boolean
        Includes various sanity checks while running the model fitting.

    Attributes
    ----------
    components_ : np.array[n_docs, n_topics]
        Document topic distribution.
    doc_topic_ : np.array[n_docs, n_topics]
        Document topic distribution.
    word_topic_ : np.array[n_vocabulary, n_topics]
        Word-topic distribution.
    """

    def __init__(
        self, n_topics, alpha=1e-1, beta=1e-1, niter=10, seed=None, verbose=False, debug=False
    ):
        self.n_topics = n_topics
        self.alpha_ = alpha
        self.beta_ = beta
        self.niter_ = niter
        self.seed_ = seed
        self.verbose_ = verbose
        self.debug_ = debug

    def _check_input(self, X):
        if not issparse(X):
            X = coo_matrix(X)
        else:
            X = X.tocoo()
        return X

    def _init(self, X):
        # init params
        mu = check_random_state(self.seed_).rand(X.nnz, self.n_topics)
        mu /= mu.sum(1, keepdims=True)
        self.mu_ = mu

    def fit(self, X):
        X = self._check_input(X)
        self._init(X)

        # prepare data
        w = X.row
        d = X.col
        x = X.data.reshape(-1, 1)
        xsumcheck = X.sum()
        n_tot_words = X.nnz

        # prepare summary statistics
        docneig_signal = np.zeros((X.shape[0], self.n_topics))
        wordneig_signal = np.zeros((X.shape[1], self.n_topics))

        update_neighs(
            self.mu_,
            docneig_signal,
            wordneig_signal,
            w,
            d,
            x.reshape(-1).astype("float"),
            self.n_topics,
            n_tot_words,
        )

        for epoch in tqdm(range(self.niter_)):
            if self.debug_:
                assert np.all(self.mu_ > 0.0), "mu not positive"
                np.testing.assert_allclose(self.mu_.sum(), X.nnz)

                np.testing.assert_allclose(docneig_signal.sum(), xsumcheck)
                np.testing.assert_allclose(wordneig_signal.sum(), xsumcheck)
                assert np.all(docneig_signal >= 0.0), "docneig_signal not positive"
                assert np.all(wordneig_signal >= 0.0), "wordneig_signal not positive"

            ret = update_mu_async(
                self.mu_,
                docneig_signal,
                wordneig_signal,
                w,
                d,
                x.reshape(-1).astype("float"),
                self.alpha_,
                self.beta_,
                self.n_topics,
                n_tot_words,
            )

            if self.verbose_:
                print("epoch {}: mean-change {}".format(epoch, ret))

        self.doc_topic_ = wordneig_signal + self.beta_
        self.word_topic_ = docneig_signal + self.alpha_
        self.doc_topic_ /= self.doc_topic_.sum(1, keepdims=True)
        self.word_topic_ /= self.word_topic_.sum(0, keepdims=True)
        return self

    def fit_transform(self, X):
        return self.fit(X).doc_topic_


#def lda_belief_propagation(
#    X, K, alpha=1e-1, beta=1e-1, niter=10, seed=None, verbose=False, debug=False
#):
#    """
#    Fits latent dirichlet allocation model using belief propagation
#    based on the mean field approximation.
#    The algorithm is based on
#    Zeng, Cheung, Liu, Learning Topic Models by belief propagation. 2012.
#    IEEE Transactions on Pattern analysis and machine learning.
#
#    Examples
#    --------
#
#    .. code-block:: python
#
#       import numpy as np
#       from bplda import lda
#
#       # toy data (10 documents, vocabulary size=5)
#       minitest = np.zeros((5, 10))
#       minitest[:3,:5]=1
#       minitest[-3:,-5:]=1
#
#       wt, dt = lda(minitest, 3, niter=10, seed=10)
#
#       print('word-topic statistics (not normalized)')
#       wt
#
#       print('document-topic statistics (not normalized)')
#       dt
#
#
#    Parameters
#    -----------
#
#    X : np.array[V, D] or sparse array
#        Word document matrix of size V x D. V denotes the vocabulary size
#        and D denotes the number of documents
#    K : int
#        Number of topics
#    alpha : float
#        Word-topic prior. Default: 0.1
#    beta : float
#        Document-topic prior. Default: 0.1
#    niter : int
#        Number updates using belief propagation.
#    seed : int or None
#        Random seed for reproducibility
#    verbose : boolean
#        Prints progress per iteration.
#    debug : boolean
#        Includes various sanity checks while running the model fitting.
#
#    Returns
#    -------
#    tuple(np.array[V, K], np.array[D, K]):
#        Word-topic statistics and document-topic statistics.
#        These values denote the parameters for the dirichlet density.
#        That is they are not normalized.
#
#    """
#    if not issparse(K):
#        X = coo_matrix(X)
#    else:
#        X = X.tocoo()
#
#    # data
#    w = X.row
#    d = X.col
#    x = X.data.reshape(-1, 1)
#    xsumcheck = X.sum()
#    N = x.shape[0]
#
#    # init params
#    mu = check_random_state(seed).rand(len(x), K)
#    mu /= mu.sum(1, keepdims=True)
#
#    docneig_signal = np.zeros((X.shape[0], K))
#    wordneig_signal = np.zeros((X.shape[1], K))
#
#    update_neighs(
#        mu, docneig_signal, wordneig_signal, w, d, x.reshape(-1).astype("float"), K, N
#    )
#
#    for epoch in tqdm(range(niter)):
#        if debug:
#            assert np.all(mu > 0.0), "mu not positive {}".format(mu)
#            np.testing.assert_allclose(mu.sum(), X.nnz)
#
#            np.testing.assert_allclose(docneig_signal.sum(), xsumcheck)
#            np.testing.assert_allclose(wordneig_signal.sum(), xsumcheck)
#            assert np.all(docneig_signal >= 0.0), "docneig_signal not positive"
#            assert np.all(wordneig_signal >= 0.0), "wordneig_signal not positive"
#
#        ret = update_mu_async(
#            mu,
#            docneig_signal,
#            wordneig_signal,
#            w,
#            d,
#            x.reshape(-1).astype("float"),
#            alpha,
#            beta,
#            K,
#            N,
#        )
#
#        # assert np.all(mu > 0.0), "mu not positive {}".format(mu_next)
#
#        if verbose:
#            print("epoch {}: mean-change {}".format(epoch, ret))
#
#    return docneig_signal + alpha, wordneig_signal + beta
