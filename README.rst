========
Overview
========



.. end-badges

This package implements Latent Dirichlet Allocation fitted via loopy belief propagation
as described in Zeng et al. Learning topic models using belief propagation. 2012, IEEE Transations on pattern analysis and machine intelligence.

The package is available under a GPL-v3 license.

Installation
============

::

    pip install bplda

You can also install the in-development version with::

    pip install https://github.com/wkopp/bplda/archive/master.zip


Getting started
===============

.. code-block:: python

       import numpy as np
       from bplda import lda

       # toy data (10 documents, vocabulary size=5)
       minitest = np.zeros((5, 10))
       minitest[:3,:5]=1
       minitest[-3:,-5:]=1

       # fit the model using 3 topics
       wt, dt = lda(minitest, 3, niter=10, seed=10)

       print('word-topic statistics (not normalized)')
       wt

       print('document-topic statistics (not normalized)')
       dt

    pip install bplda

Documentation
=============

https://bplda.readthedocs.io/


