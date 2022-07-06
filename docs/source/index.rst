Nyxus
===============

Nyxus is a feature-rich, highly optimized, Python/C++ application capable of analyzing images of arbitrary size 
and assembling complex regions of interest (ROIs) split across multiple image tiles and files. This accomplished through 
multi-threaded tile prefetching and a three phase analysis pipeline shown below.

.. image :: nyxus_workflow.jpg

Nyxus can be used via Python or command line and is available in containerized form for reproducible execution. 
Nyxus computes over 450 combined intensity, texture, and morphological features at the ROI or whole image level with 
more in development. Key features that make Nyxus unique among other image feature extraction applications is its ability 
to operate at any scale, its highly validated algorithms, and its modular nature that makes the addition of new features straightforward.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Math
   devguide
   examples
   featurelist
   References
   