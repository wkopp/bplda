import numpy as np
from scipy.sparse import csr_matrix
from bplda import LDA


def test_lda2():

    # toy data (10 documents, vocabulary size=5)
    minitest = np.zeros((5, 10))
    minitest[:3, :5] = 1
    minitest[-3:, -5:] = 1

    lda = LDA(3, niter=100, seed=10)
    lda.fit(minitest)
    wt = lda.word_topic_
    dt = lda.doc_topic_

    print("word-topic statistics (not normalized)")
    wt

    print("document-topic statistics (not normalized)")
    dt
    compwt = np.asarray(
            [
                [0.10706605, 2.59646734, 2.59646661],
                [0.10706605, 2.59646732, 2.59646664],
                [4.81434035, 2.74283003, 2.74282962],
                [5.067708, 0.116146, 0.116146],
                [5.067708, 0.116146, 0.116146],
            ]
        )
    compwt /= compwt.sum(0, keepdims=True)
    np.testing.assert_allclose(
        wt, compwt
    )
    compdt = np.asarray(
            [
                [0.14330033, 1.57835006, 1.5783496],
                [0.14330033, 1.57835017, 1.5783495],
                [0.14330033, 1.57835017, 1.5783495],
                [0.14330033, 1.57834998, 1.57834968],
                [0.14330033, 1.5783497, 1.57834996],
                [2.98947735, 0.15526132, 0.15526133],
                [2.98947735, 0.15526132, 0.15526133],
                [2.98947735, 0.15526132, 0.15526133],
                [2.98947735, 0.15526132, 0.15526133],
                [2.98947735, 0.15526132, 0.15526133],
            ]
        )
    compdt /= compdt.sum(1, keepdims=True)
    np.testing.assert_allclose(
        dt, compdt
    )

def test_lda3():

    # toy data (10 documents, vocabulary size=5)
    minitest = np.zeros((5, 10))
    minitest[:3, :5] = 1
    minitest[-3:, -5:] = 1

    lda = LDA(3, niter=100, seed=10, verbose=True, debug=True)
    dt = lda.fit_transform(csr_matrix(minitest))

    lda.fit(csr_matrix(minitest))
    wt = lda.word_topic_

    print("word-topic statistics (not normalized)")
    wt

    print("document-topic statistics (not normalized)")
    dt
    compwt = np.asarray(
            [
                [0.10706605, 2.59646734, 2.59646661],
                [0.10706605, 2.59646732, 2.59646664],
                [4.81434035, 2.74283003, 2.74282962],
                [5.067708, 0.116146, 0.116146],
                [5.067708, 0.116146, 0.116146],
            ]
        )
    compwt /= compwt.sum(0, keepdims=True)
    np.testing.assert_allclose(
        wt, compwt
    )
    compdt = np.asarray(
            [
                [0.14330033, 1.57835006, 1.5783496],
                [0.14330033, 1.57835017, 1.5783495],
                [0.14330033, 1.57835017, 1.5783495],
                [0.14330033, 1.57834998, 1.57834968],
                [0.14330033, 1.5783497, 1.57834996],
                [2.98947735, 0.15526132, 0.15526133],
                [2.98947735, 0.15526132, 0.15526133],
                [2.98947735, 0.15526132, 0.15526133],
                [2.98947735, 0.15526132, 0.15526133],
                [2.98947735, 0.15526132, 0.15526133],
            ]
        )
    compdt /= compdt.sum(1, keepdims=True)
    np.testing.assert_allclose(
        dt, compdt
    )
