.. -*- rst -*-

Asynchronous Sigma-Delta Modulator Algorithms
=============================================

Single-Input Single-Output Algorithms
-------------------------------------

:func:`Time Encoding Machine <bionet.ted.asdm.asdm_encode>` |lazar_perfect_2004|_
   Encodes a bandlimited signal using an Asynchronous Sigma-Delta
   Modulator.

   .. image:: images/tem-asdm.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine <bionet.ted.asdm.asdm_decode>` |lazar_perfect_2004|_
   Reconstructs a bandlimited signal encoded with an Asynchronous Sigma-Delta
   Modulator using sinc kernels.

   .. image:: images/tdm-sinc.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine - Fast Approximation Method <bionet.ted.asdm.asdm_decode_fast>` |lazar_fast_2005|_
   Reconstructs a bandlimited signal encoded with an Asynchronous Sigma-Delta
   Modulator using a fast approximation method.

   .. image:: images/tdm-fast.png
      :scale: 70
      :align: center

:class:`Time Decoding Machine - Real-Time Decoder <bionet.ted.rt.ASDMRealTimeDecoder>` |lazar_overcomplete_2008|_
   Decodes a bandlimited, arbitrarily long signal encoded by an
   Asynchronous Sigma-Delta Modulator by stitching together blocks of data
   decoded by :func:`solving a Vandermonde system <bionet.ted.asdm.asdm_decode_vander>` using the
   :func:`Björk-Pereyra Algorithm <bionet.ted.bpa.bpa>`.

   .. image:: images/tdm-real.png
      :scale: 70
      :align: center

:class:`Time Decoding Machine - Threshold-Insensitive Real-Time Decoder <bionet.ted.rt.ASDMRealTimeDecoder>` |lazar_overcomplete_2008|_
   Decodes a bandlimited, arbitrarily long signal encoded by an
   Asynchronous Sigma-Delta Modulator by stitching together blocks of data
   decoded by :func:`solving a Vandermonde system <bionet.ted.asdm.asdm_decode_vander_ins>` using the
   :func:`Björk-Pereyra Algorithm <bionet.ted.bpa.bpa>`. This
   reconstruction method does not require the specification of an
   integrator threshold.

   .. image:: images/tdm-real-ins.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine - Threshold-Insensitive Method <bionet.ted.asdm.asdm_decode_ins>` |lazar_perfect_2004|_
   Reconstructs a bandlimited signal encoded with an Asynchronous
   Sigma-Delta Modulator using sinc kernels. This reconstruction
   method does not require the specification of an integrator
   threshold.

   .. image:: images/tdm-sinc-ins.png
      :scale: 70
      :align: center

Multiple-Input Single-Output Algorithms
---------------------------------------

:func:`Time Decoding Machine - MISO Decoder <bionet.ted.asdm.asdm_decode_pop>` |lazar_information_2007|_
   Decodes a bandlimited signal encoded by multiple Asynchronous
   Sigma-Delta Modulators using sinc kernels.

   .. image:: images/tdm-sinc-miso.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine - Threshold-Insensitive MISO Decoder <bionet.ted.asdm.asdm_decode_pop>` |lazar_perfect_2004|_ |lazar_information_2007|_
   Decodes a bandlimited signal encoded by multiple Asynchronous
   Sigma-Delta Modulators using sinc kernels. This reconstruction
   method does not require the specification of an integrator
   thresholds.

   .. image:: images/tdm-sinc-ins-miso.png
      :scale: 70
      :align: center

.. include:: bibliography.rst
