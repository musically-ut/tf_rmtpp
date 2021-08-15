tf_rmtpp
========

.. image:: https://img.shields.io/pypi/v/tf_rmtpp.svg
    :target: https://pypi.python.org/pypi/tf_rmtpp
    :alt: Latest PyPI version

.. image::  .png
   :target:
   :alt: Latest Travis CI build status

An (unofficial) implementation of `Recurrent Marked Temporal Point Processes <https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf>`_ in TensorFlow.

Citation:

    Du, N., Dai, H., Trivedi, R., Upadhyay, U., Gomez-Rodriguez, M., & Song, L. (2016, August). Recurrent marked temporal point processes: Embedding event history to vector. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1555-1564). ACM.
   

Usage
-----

Installation
------------

.. code:: bash

    pip install git+https://github.com/musically-ut/tf_rmtpp.git@master#egg=tf_rmtpp

Requirements
^^^^^^^^^^^^

  - ``TensorFlow``
  - ``pandas``

Compatibility
-------------

 - ``Python 3``

Licence
-------

MIT

Authors
-------

`tf_rmtpp` was written by `Utkarsh Upadhyay <musically.ut@gmail.com>`_.

Other implementations
---------------------

Checkout `woshiyyya/ERPP-RMTPP <https://github.com/woshiyyya/ERPP-RMTPP>`_ for an implementation based on `PyTorch`.
