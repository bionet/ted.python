.. -*- rst -*-

Integrate-and-Fire Neuron Algorithms
====================================

Single-Input Single-Output Algorithms
-------------------------------------


:func:`Time Encoding Machine <bionet.ted.iaf.iaf_encode>` |lazar_perfect_2004|_
   Encodes a bandlimited signal using an Integrate-and-Fire neuron.
   Leaky and ideal neuron models are supported.

   .. image:: images/tem-iaf.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine <bionet.ted.iaf.iaf_decode>` |lazar_time_2004|_
   Decodes a bandlimited signal encoded by an Integrate-and-Fire
   neuron using sinc kernels.

   .. image:: images/tdm-sinc.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine - Fast Approximation Method <bionet.ted.iaf.iaf_decode_fast>` |lazar_fast_2005|_
   Decodes a bandlimited signal encoded by an Integrate-and-Fire
   neuron using a fast approximation method.

   .. image:: images/tdm-fast.png
      :scale: 70
      :align: center

:class:`Time Decoding Machine - Real-Time Decoder <bionet.ted.rt.IAFRealTimeDecoder>` |lazar_overcomplete_2008|_
   Decodes a bandlimited, arbitrarily long signal encoded by an
   Integrate-and-Fire neuron by stitching together blocks of data
   decoded by :func:`solving a Vandermonde system <bionet.ted.iaf.iaf_decode_vander` using the
   :func:`Björk-Pereyra Algorithm <bionet.ted.bpa.bpa>`.

   .. image:: images/tdm-real.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine - Spline Interpolation Method <bionet.ted.iaf.iaf_decode_spline>` |lazar_consistent_2009|_
   Decodes a bandlimited signal encoded by an Integrate-and-Fire
   neuron using spline interpolation.

   .. image:: images/tdm-spline.png
      :scale: 70
      :align: center

Single-Input Multiple-Output Algorithms
---------------------------------------

:func:`Time Encoding Machine - SIMO Coupled IAF Encoder <bionet.ted.iaf.iaf_encode_coupled>` |lazar_consistent_2009|_
   Encodes a finite energy signal encoded by with multiple coupled
   ON-OFF Integrate-and-Fire neurons.

   .. image:: images/tem-iaf-coupled.png
      :scale: 70
      :align: center

Multiple-Input Single-Output Algorithms
---------------------------------------

:func:`Time Decoding Machine - MISO IAF Decoder <bionet.ted.iaf.iaf_decode_pop>` |lazar_information_2007|_
   Decodes a bandlimited signal encoded by multiple Integrate-and-Fire
   neurons using sinc kernels.

   .. image:: images/tdm-sinc-miso.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine - MISO Coupled IAF Decoder <bionet.ted.iaf.iaf_decode_coupled>` |lazar_consistent_2009|_
   Decodes a finite energy signal encoded by multiple coupled ON-OFF
   Integrate-and-Fire neurons using spline interpolation.

   .. image:: images/tdm-spline-mimo.png
      :scale: 70
      :align: center

Multiple-Input Multiple-Output Algorithms
-----------------------------------------

:func:`Time Encoding Machine - MIMO Delayed IAF Encoder <bionet.ted.iaf.iaf_encode_delay>` |lazar_consistent_2009|_
   Encodes several finite energy signals encoded by multiple
   Integrate-and-Fire neurons with delays.

   .. image:: images/tem-iaf-mimo.png
      :scale: 70
      :align: center

:func:`Time Decoding Machine - MIMO Delayed IAF Decoder <bionet.ted.iaf.iaf_decode_delay>` |lazar_consistent_2009|_
   Reconstructs several finite energy signals encoded by multiple
   Integrate-and-Fire neurons with delays using spline
   interpolation.

   .. image:: images/tdm-spline-mimo.png
      :scale: 70
      :align: center

.. include:: bibliography.rst
