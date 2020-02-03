import numpy as np
from scipy.sparse import csr_matrix
from bplda import LDA


def test_lda2():

    # toy data (10 documents, vocabulary size=5)
    minitest = np.zeros((5, 10))
    minitest[:3, :5] = 1
    minitest[-3:, -5:] = 1

    lda = LDA(3, niter=10, seed=10)
    lda.fit(minitest)
    wt = lda.word_topic_
    dt = lda.doc_topic_

    print("word-topic statistics (not normalized)")
    wt

    print("document-topic statistics (not normalized)")
    dt
    compwt = np.asarray(
        [
            [0.00708602, 0.3310797, 0.29493918],
            [0.007086, 0.32899726, 0.29853138],
            [0.31779248, 0.31831982, 0.36517315],
            [0.33401681, 0.01080145, 0.02068081],
            [0.3340187, 0.01080176, 0.02067548],
        ]
    )

    np.testing.assert_allclose(wt, compwt, atol=1e-6, rtol=1e-6)
    compdt = np.asarray(
        [
            [0.04393251, 0.60705708, 0.34901041],
            [0.04393614, 0.60896743, 0.34709643],
            [0.0439667, 0.62515048, 0.33088282],
            [0.04393936, 0.60818744, 0.3478732],
            [0.04394717, 0.60986784, 0.34618499],
            [0.9052288, 0.04527242, 0.04949878],
            [0.90524182, 0.04527273, 0.04948544],
            [0.90525461, 0.04527307, 0.04947232],
            [0.9052672, 0.04527343, 0.04945937],
            [0.90527961, 0.04527381, 0.04944658],
        ]
    )

    np.testing.assert_allclose(dt, compdt, atol=1e-6, rtol=1e-6)


def test_lda3():

    # toy data (10 documents, vocabulary size=5)
    minitest = np.zeros((5, 10))
    minitest[:3, :5] = 1
    minitest[-3:, -5:] = 1

    lda = LDA(3, niter=10, seed=10, verbose=True, debug=True)
    dt = lda.fit_transform(csr_matrix(minitest))

    lda.fit(csr_matrix(minitest))
    wt = lda.word_topic_

    print("word-topic statistics (not normalized)")
    wt

    print("document-topic statistics (not normalized)")
    dt
    compwt = np.asarray(
        [
            [0.00708602, 0.3310797, 0.29493918],
            [0.007086, 0.32899726, 0.29853138],
            [0.31779248, 0.31831982, 0.36517315],
            [0.33401681, 0.01080145, 0.02068081],
            [0.3340187, 0.01080176, 0.02067548],
        ]
    )

    np.testing.assert_allclose(wt, compwt, atol=1e-6, rtol=1e-6)
    compdt = np.asarray(
        [
            [0.04393251, 0.60705708, 0.34901041],
            [0.04393614, 0.60896743, 0.34709643],
            [0.0439667, 0.62515048, 0.33088282],
            [0.04393936, 0.60818744, 0.3478732],
            [0.04394717, 0.60986784, 0.34618499],
            [0.9052288, 0.04527242, 0.04949878],
            [0.90524182, 0.04527273, 0.04948544],
            [0.90525461, 0.04527307, 0.04947232],
            [0.9052672, 0.04527343, 0.04945937],
            [0.90527961, 0.04527381, 0.04944658],
        ]
    )

    np.testing.assert_allclose(dt, compdt, atol=1e-6, rtol=1e-6)
