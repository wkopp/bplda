========
Overview
========

.. start-badges

.. end-badges

This package implements Latent Dirichlet Allocation fitted via loopy belief propagation
as described in Zeng et al. Learning topic models using belief propagation. 2012, IEEE Transations on pattern analysis and machine intelligence.

The package is available under a GPL-v3 license.

Installation
============

You can also install the package with

::

    pip install https://github.com/wkopp/bplda/archive/master.zip


Getting started
===============

.. code-block:: python

   import numpy as np
   from bplda import LDA

   # toy data (10 documents, vocabulary size=5)
   minitest = np.zeros((5, 10))
   minitest[:3,:5]=1
   minitest[-3:,-5:]=1

   model = LDA(3, niter=10, seed=10)

   # fit and get document-topic matrix
   doc_top = model.fit_transform(minitest)

   # access word-topic matrix
   model.word_topic_

   # compute the log-likelihood score
   model.score(minitest)

   model.loglikeli_
