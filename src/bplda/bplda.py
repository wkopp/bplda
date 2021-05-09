"""Main module."""
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import coo_matrix
from sklearn.utils import check_random_state
from bplda._bplda import update_neighs
from bplda._bplda import update_mu_async
from bplda._bplda import _marginal_loglikelihood
from bplda._bplda import collapsed_gibbs_sampling
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import loggamma


class BeliefPropLDA(BaseEstimator, TransformerMixin):
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
       X = np.zeros((5, 10))
       X[:3,:5]=1
       X[-3:,-5:]=1

       model = LDA(3, niter=10, seed=10)
       topic_doc = model.fit_transform(X)

       model.word_topic_prob_


    Parameters
    -----------

    n_topics : int
        Number of topics
    alpha : float
        Document-topic prior. Default: 0.1
    beta : float
        Word-topic prior. Default: 0.1
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
    topic_doc_prob_ : np.array[n_topics, n_docs]
        Topic-document distribution.
    word_topic_prob_ : np.array[n_vocabulary, n_topics]
        Word-topic distribution.
    """

    def __init__(
        self,
        n_topics,
        alpha=1e-1,
        beta=1e-1,
        niter=10,
        seed=None,
        verbose=False,
        debug=False,
        evaluate_every=100
    ):
        self.num_topics = self.n_topics = n_topics
        self.alpha_ = alpha
        self.beta_ = beta
        self.niter_ = niter
        self.seed_ = seed
        self.verbose_ = verbose
        self.debug_ = debug
        self.evaluate_every = evaluate_every

    def _init(self, X):
        # init params
        mu = check_random_state(self.seed_).rand(X.nnz, self.n_topics)
        mu /= mu.sum(1, keepdims=True)
        self.mu_ = mu

    def _check_input(self, X):
        if not issparse(X):
            X = coo_matrix(X)
        else:
            X = X.tocoo()
        return X

    def fit(self, X):
        """ Fit the model for X with belief propagation.

        Parameters
        ----------
        X : np.array[n_vocabulary, n_docs]
            Word-document data matrix.

        Returns
        -------
        self

        """

        X = self._check_input(X)
        self._init(X)
        self.history = []

        # prepare data
        w = X.row
        d = X.col
        x = X.data.reshape(-1, 1)
        xsumcheck = X.sum()
        n_tot_words = X.nnz

        # prepare summary statistics
        self.word_topic_matrix = np.zeros((X.shape[0], self.n_topics))
        self.topic_document_matrix = np.zeros((self.n_topics, X.shape[1]))

        update_neighs(
            self.mu_,
            self.word_topic_matrix,
            self.topic_document_matrix,
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

                np.testing.assert_allclose(self.word_topic_matrix.sum(), xsumcheck)
                np.testing.assert_allclose(self.topic_document_matrix.sum(), xsumcheck)
                assert np.all(self.word_topic_matrix >= 0.0), "word_topic_matrix not positive"
                assert np.all(self.topic_document_matrix >= 0.0), "topic_document_matrix not positive"

            ret = update_mu_async(
                self.mu_,
                self.word_topic_matrix,
                self.topic_document_matrix,
                w,
                d,
                x.reshape(-1).astype("float"),
                self.alpha_,
                self.beta_,
                self.n_topics,
                n_tot_words,
            )

            if (epoch % self.evaluate_every) == 0:
                self.topic_doc_prob_ = self.topic_document_matrix + self.alpha_
                self.word_topic_prob_ = self.word_topic_matrix + self.beta_
                #self.topic_doc_prob_ /= self.topic_doc_prob_.sum(0, keepdims=True)
                #self.word_topic_prob_ /= self.word_topic_prob_.sum(0, keepdims=True)
                self.loglikeli_ = self.score(X)
                self.history.append(self.score(X))
                #print(
                #    "epoch {}: mean-change={}, loglikeli={}".format(
                #        epoch, ret, self.loglikeli_
                #    )
                #)

        self.topic_doc_prob_ = self.topic_document_matrix + self.alpha_
        self.word_topic_prob_ = self.word_topic_matrix + self.beta_
        self.topic_doc_prob_ /= self.topic_doc_prob_.sum(0, keepdims=True)
        self.word_topic_prob_ /= self.word_topic_prob_.sum(0, keepdims=True)
        self.history.append(self.score(X))
        self.loglikeli_ = self.history[-1]
        return self

    def fit_transform(self, X):
        """ Fit the model for X with belief propagation and returns transformation.

        Parameters
        ----------
        X : np.array[n_vocabulary, n_docs]
            Word-document data matrix.

        Returns
        -------
        topic_doc : np.array[n_topics, n_docs]
            Topic-document matrix
        """
        return self.fit(X).topic_doc_prob_

    def score(self, X):
        """ Likelihood of the model

        Parameters
        ----------
        X : np.array[n_vocabulary, n_docs]
            Word-document data matrix.

        Returns
        -------
        loglikelihood : float
        """
        if not issparse(X):
            X = coo_matrix(X)
        else:
            X = X.tocoo()

        # data
        w = X.row
        d = X.col
        x = X.data.reshape(-1, 1)

        assert X.shape[0] == self.word_topic_prob_.shape[0]
        assert X.shape[1] == self.topic_doc_prob_.shape[1]

        loglikeli = self.num_topics * loggamma(self.beta_ * X.shape[0])
        loglikeli -= self.num_topics * X.shape[0] * loggamma(self.beta_)
        loglikeli += X.shape[1] * loggamma(self.alpha_*self.num_topics)
        loglikeli -= X.shape[1] * self.num_topics * loggamma(self.alpha_)

        loglikeli += loggamma(self.word_topic_matrix + self.beta_).sum()
        loglikeli -= loggamma(self.word_topic_matrix.sum(0) + self.beta_*X.shape[0]).sum()
        loglikeli += loggamma(self.topic_document_matrix + self.alpha_).sum()
        loglikeli -= loggamma(self.topic_document_matrix.sum(0) + self.alpha_*self.num_topics).sum()
        return loglikeli
        #l1 = loggamma(self.word_topic_matrix + self.beta_) - loggamma(self.word_topic_matrix.sum(0, keepdims=True) + self.beta_*X.shape[0])
        #l2 = loggamma(self.topic_document_matrix + self.alpha_) - loggamma(self.topic_document_matrix.sum(0, keepdims=True) + self.alpha_*self.num_topics)

        #return _marginal_loglikelihood(loglikeli, l1, l2, X.shape[0], X.shape[1], self.num_topics)

    def perplexity(self, X):
        pass

class CollapsedGibbsLDA(BaseEstimator, TransformerMixin):
    """
    Latent dirichlet allocation model with collapsed Gibbs sampling.
    Griffith and Steyvers, PNAS, 2004.

    Examples
    --------

    .. code-block:: python

       import numpy as np
       from bplda import CollapsedGibbsLDA

       # toy data (10 documents, vocabulary size=5)
       X = np.zeros((5, 10))
       X[:3,:5]=1
       X[-3:,-5:]=1

       model = LDA(3, niter=10, seed=10)
       topic_doc = model.fit_transform(X)

       model.word_topic_


    Parameters
    -----------

    n_topics : int
        Number of topics
    alpha : float
        Document-topic prior. Default: 0.1
    beta : float
        Word-topic prior. Default: 0.1
    burnin : int
        Number of burn-in iterations.
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
    topic_doc_prob_ : np.array[n_topics, n_docs]
        Document topic distribution.
    word_topic_prob_ : np.array[n_vocabulary, n_topics]
        Word-topic distribution.
    """

    def __init__(
        self,
        n_topics,
        alpha=1e-1,
        beta=1e-1,
        burnin=100,
        niter=100,
        seed=None,
        verbose=False,
        debug=False,
        evaluate_every=100
    ):
        self.num_topics = n_topics
        self.alpha_ = alpha
        self.beta_ = beta
        self.burnin_ = burnin
        self.niter_ = niter
        self.seed_ = seed
        self.verbose_ = verbose
        self.debug_ = debug
        self.evaluate_every = evaluate_every

    def _check_input(self, X):
        if not issparse(X):
            X = coo_matrix(X)
        else:
            X = X.tocoo()
        return X.astype('float')

    def unpack_data(self, X):
        X = self._check_input(X)
        num_words, num_documents = X.shape
        word_multiples = X.data
        observed_words = X.row
        observed_documents = X.col
        num_words, num_documents, word_multiples, observed_words, observed_documents

    def _init(self, X):

        X = self._check_input(X)
        self.num_words, self.num_documents = X.shape
        self.word_multiples = X.data
        self.observed_words = X.row
        self.observed_documents = X.col
        # init params
        z = check_random_state(self.seed_).randint(0, self.num_topics, size=X.nnz).astype('int32')
        self.topic_assignments = z
        self.word_topic_matrix = np.asarray(coo_matrix((self.word_multiples, (self.observed_words, self.topic_assignments)), shape=(self.num_words, self.num_topics)).todense())
        self.topic_document_matrix = np.asarray(coo_matrix((self.word_multiples, (self.topic_assignments, self.observed_documents)), shape=(self.num_topics, self.num_documents)).todense())
        self.document_counts = np.asarray(self.topic_document_matrix.sum(0)).flatten()
        self.topic_counts = np.asarray(self.topic_document_matrix.sum(1)).flatten()

    def prepare_data(self, X):
        X = self._check_input(X)
        num_words, num_documents = X.shape
        word_multiples = X.data
        observed_words = X.row
        observed_documents = X.col
        # init params
        z = check_random_state(self.seed_).randint(0, self.num_topics, size=X.nnz).astype('int32')

        word_topic_matrix = np.asarray(coo_matrix((word_multiples, (observed_words, z)), shape=(num_words, self.num_topics)).todense())
        topic_document_matrix = np.asarray(coo_matrix((word_multiples, (z, observed_documents)), shape=(self.num_topics, num_documents)).todense())
        document_counts = np.asarray(topic_document_matrix.sum(0)).flatten()
        topic_counts = np.asarray(topic_document_matrix.sum(1)).flatten()

        return num_words, num_documents, observed_words, \
               observed_documents, word_multiples, z, \
               word_topic_matrix, topic_document_matrix, document_counts, \
               topic_counts

    def debug(self):
        if not self.debug_:
            return
        n=self.word_multiples.sum()
        np.testing.assert_allclose(n, self.word_topic_matrix.sum())
        np.testing.assert_allclose(n, self.topic_document_matrix.sum())
        np.testing.assert_allclose(n, self.document_counts.sum())
        np.testing.assert_allclose(n, self.topic_counts.sum())

    def fit(self, X):
        """ Fit the model for X with belief propagation.

        Parameters
        ----------
        X : np.array[n_vocabulary, n_docs]
            Word-document data matrix.

        Returns
        -------
        self

        """

        self.history = []
        rand_state = check_random_state(self.seed_)
        X = self._check_input(X)
        self._init(X)

        for epoch in tqdm(range(self.burnin_)):
            self.debug()
            ret = collapsed_gibbs_sampling(
                self.observed_words,
                self.observed_documents,
                self.word_multiples,
                self.topic_assignments,
                self.word_topic_matrix,
                self.topic_document_matrix,
                rand_state.rand(len(self.observed_words)),
                len(self.observed_words),
                int(self.num_topics),
                int(self.num_words),
                self.alpha_,
                self.beta_,
                self.document_counts,
                self.topic_counts,
                0
            )
            if (epoch % self.evaluate_every) == 0:
                 self.history.append(self.score(X))

        for epoch in tqdm(range(self.niter_)):
            self.debug()
            ret = collapsed_gibbs_sampling(
                self.observed_words,
                self.observed_documents,
                self.word_multiples,
                self.topic_assignments,
                self.word_topic_matrix,
                self.topic_document_matrix,
                rand_state.rand(len(self.observed_words)),
                len(self.observed_words),
                int(self.num_topics),
                int(self.num_words),
                self.alpha_,
                self.beta_,
                self.document_counts,
                self.topic_counts,
                0
            )

            if (epoch % self.evaluate_every) == 0:
                 self.history.append(self.score(X))

        self.topic_doc_prob_ = self.topic_document_matrix + self.alpha_
        self.word_topic_prob_ = self.word_topic_matrix + self.beta_
        self.topic_doc_prob_ /= self.topic_doc_prob_.sum(0, keepdims=True)
        self.word_topic_prob_ /= self.word_topic_prob_.sum(0, keepdims=True)
        self.history.append(self.score(X))
        self.loglikeli_ = self.history[-1]
        return self

    def fit_transform(self, X):
        """ Fit the model for X with belief propagation and returns transformation.

        Parameters
        ----------
        X : np.array[n_vocabulary, n_docs]
            Word-document data matrix.

        Returns
        -------
        topic_doc : np.array[n_topics, n_docs]
            Topic-document matrix
        """
        return self.fit(X).topic_doc_prob_

    def score(self, X=None):
        """ Likelihood of the model

        Parameters
        ----------
        X : np.array[n_vocabulary, n_docs]
            Word-document data matrix.

        Returns
        -------
        loglikelihood : float
        """
        if X is None:
            nw = self.num_words
            nd = self.num_documents
            wtm = self.word_topic_matrix
            tdm = self.topic_document_matrix
        else:
            tdm = self._transform(X)

        loglikeli = self.num_topics*loggamma(self.beta_*self.num_words)
        loglikeli -= self.num_topics*self.num_words*loggamma(self.beta_)
        loglikeli += self.num_documents * loggamma(self.alpha_*self.num_topics)
        loglikeli -= self.num_documents * self.num_topics * loggamma(self.alpha_)

        loglikeli += loggamma(self.word_topic_matrix + self.beta_).sum()
        loglikeli -= loggamma(self.word_topic_matrix.sum(0) + self.beta_*self.num_words).sum()
        loglikeli += loggamma(tdm + self.alpha_).sum()
        loglikeli -= loggamma(tdm.sum(0) + self.alpha_*self.num_topics).sum()
        return loglikeli

    def transform(self, X=None):
        if X is None:
            return self.topic_document_matrix
        return self._transform(X)

    def _transform(self, X):
        nw, nd, ow, od, wm, z,  wtm, tdm, dc, tc = self.prepare_data(X)
        assert self.num_words == nw

        rand_state = check_random_state(self.seed_)
        for epoch in range(3):
            self.debug()
            ret = collapsed_gibbs_sampling(
                ow,
                od,
                wm,
                z,
                self.word_topic_matrix,
                tdm,
                rand_state.rand(len(ow)),
                len(ow),
                int(self.num_topics),
                int(self.num_words),
                self.alpha_,
                self.beta_,
                dc,
                tc,
                1
            )
        return tdm

