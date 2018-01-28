Tensor Wrapper
--------------

The tensor wrapper is a data structure which mimics the behavior of a `numpy.ndarray <http://docs.scipy.org/doc/numpy/index.html>`_ and associates each item of the tensor with the evaluation of a user-defined function on the corresponding grid point.

Let for example :math:`\mathcal{X} = \times_{i=1}^d {\bf x}_i`, where :math:`{\bf x}_i` define the position of the grid points in the :math:`i`-th direction. Let us consider the function :math:`f:\mathcal{X}\rightarrow \mathbb{R}^{n_1\times \ldots \times n_m}`. Let us define the tensor valued tensor :math:`\mathcal{A}=f(\mathcal{X})`. Thus any entry :math:`\mathcal{A}[i_1,\ldots,i_d] = f({\bf x}_{i_1},\ldots,{\bf x}_{i_d})` is a tensor in :math:`\mathbb{R}^{n_1\times \ldots \times n_m}`. The storage of the whole tensor :math:`\mathcal{A}` can be problematic for big :math:`d` and :math:`m`, and not necessary if one is just willing to sample values from it. 

The :py:class:`~TensorToolbox.TensorWrapper` allows the access to the elements of :math:`\mathcal{A}` which however are not all allocated, but computed on-the-fly and stored in a hash-table data structure (a Python dictionary). The :py:class:`~TensorToolbox.TensorWrapper` can be reshaped and accessed as if it was a `numpy.ndarray <http://docs.scipy.org/doc/numpy/index.html>`_ (including slicing of indices). Additionally it allows the existence of multiple views of the tensor, sharing among them the allocated data, and it allows the *Quantics* folding used within the *Quantics Tensor Train* :cite:`Khoromskij2011,Khoromskij2010` routines :py:class:`~TensorToolbox.QTTvec`.

In the following we will use a simple example to show the capabilities of this data structure. We will let :math:`d=2` and :math:`f:\mathcal{X}\rightarrow \mathbb{R}`.

Construction
^^^^^^^^^^^^

In order to **construct** a :py:class:`~TensorToolbox.TensorWrapper` we need first to define a grid and a function.

.. doctest:: tw

   >>> import numpy as np
   >>> import itertools
   >>> import TensorToolbox as TT
   >>> d = 2
   >>> x_fine = np.linspace(-1,1,7)
   >>> params = {'k': 1.}
   >>> def f(X,params):
   >>>     return np.max(X) * params['k']
   >>> TW = TT.TensorWrapper( f, [ x_fine ]*d, params, dtype=float )

Access and data
^^^^^^^^^^^^^^^

The :py:class:`~TensorToolbox.TensorWrapper` can then be **accessed** as a `numpy.ndarray <http://docs.scipy.org/doc/numpy/index.html>`_:

.. doctest:: tw

   >>> TW[1,2]
   -0.33333333333333337

This access to the :py:class:`~TensorToolbox.TensorWrapper` has caused the evaluation of the function :math:`f` and the storage of the associated value. In order to check the **fill level** of the :py:class:`~TensorToolbox.TensorWrapper`, we do:

.. doctest:: tw

   >>> TW.get_fill_level()
   1

The **evaluation indices** at which the function has been evaluated can be retrived this way:

.. doctest:: tw

   >>> TW.get_fill_idxs()
   [(1, 2)]

The :py:class:`~TensorToolbox.TensorWrapper` can be accessed using also **slicing** along some of the coordinates:

.. doctest:: tw

   >>> TW[:,1:6:2]
   array([[-0.66666666666666674, 0.0, 0.66666666666666652],
       [-0.66666666666666674, 0.0, 0.66666666666666652],
       [-0.33333333333333337, 0.0, 0.66666666666666652],
       [0.0, 0.0, 0.66666666666666652],
       [0.33333333333333326, 0.33333333333333326, 0.66666666666666652],
       [0.66666666666666652, 0.66666666666666652, 0.66666666666666652],
       [1.0, 1.0, 1.0]], dtype=object)

The **data** already computed are stored in the dictionary :py:attr:`TensorWrapper.data`, which one can access and modify at his/her own risk. The data can be **erased** just by resetting the :py:attr:`TensorWrapper.data` field:

.. doctest:: tw

   >>> TW.data = {}

The constructed :py:class:`~TensorToolbox.TensorWrapper` to which has not been applied any of the view/extension/reshaping functions presented in the following, is called the **global** tensor wrapper. The shape informations regarding the global wrapper can be *always* accessed by:

.. doctest:: tw

   >>> TW.get_global_shape()
   (7, 7)
   >>> TW.get_global_ndim()
   2
   >>> TW.get_global_size()
   49

If no view/extension/reshaping has been applied to the :py:class:`~TensorToolbox.TensorWrapper`, then the same output is obtained by:

.. doctest:: tw

   >>> TW.get_shape()
   (7, 7)
   >>> TW.get_ndim()
   2
   >>> TW.get_size()
   49

or by 

.. doctest:: tw

   >>> TW.shape
   (7, 7)
   >>> TW.ndim
   2
   >>> TW.size
   49

.. note:: If any view/extension/reshape has been applied to the :py:class:`~TensorToolbox.TensorWrapper`, then the output of :py:meth:`TensorWrapper.get_global_shape` and :py:meth:`TensorWrapper.get_shape` will differ. Anyway :py:meth:`TensorWrapper.get_global_shape` will *always* return the information regarding the **global** tensor wrapper.

Views
^^^^^

The :py:class:`~TensorToolbox.TensorWrapper` allows the definition of multiple views over the defined tensor. The information regarding each view are contained in the dictionary :py:attr:`TensoWrapper.maps`. The main view is called ``full`` and is defined at construction time. Additional views can be defined through the function :py:meth:`TensorWrapper.set_view`. Let's continue the previous example, by adding a new view to the wrapper with a coarser grid.

.. doctest:: tw

   >>> x_coarse = np.linspace(-1,1,4)
   >>> TW.set_view( 'coarse', [x_coarse]*d )

.. note:: The grid of the ``full`` view must contain the grids associated to the new view.

The different views can be accessed separately, but they all refer to the same global data structure. In order to access the :py:class:`~TensorToolbox.TensorWrapper` through one of its views, the view must be **activated**:

.. doctest:: tw

   >>> TW.set_active_view('coarse')
   >>> TW[2,:]
   >>> TW.set_active_view('full')
   >>> TW[1,:]
   >>> TW[:,2]

The following figure shows the global grid as well as its two views, the ``full`` and the ``coarse`` views. The allocated indicies are also highlighted.

.. figure:: _static/Figures/TensorWrapperViews.*

   The global tensor and two of its views. The ``full`` view corresponds by default to the global tensor. The ``coarse`` is contained in the ``full`` view. The uniquely allocated values of the tensor are shown in the different views.

The shape characteristics of the active view can be accessed through :py:meth:`TensorWrapper.get_view_shape` and the corresponding commands for ``ndim`` and ``size``. For example:

.. doctest:: tw

   >>> TW.set_active_view('full')
   >>> TW.get_view_shape()
   (7, 7)
   >>> TW.get_shape()
   (7, 7)
   >>> TW.set_active_view('coarse')
   >>> TW.get_global_shape()
   (7, 7)
   >>> TW.get_view_shape()
   (4, 4)
   >>> TW.get_shape()
   (4, 4)
   >>> TW.shape
   (4, 4)

Grid refinement
^^^^^^^^^^^^^^^

The *global* grid can be refined using the function :py:meth:`TensorWrapper.refine`, provinding a grid which contains the previous one. This refinement does not alter the allocated data which is instead preserved and mapped to the new mesh. 

.. doctest:: tw
   
   >>> x_ffine = np.linspace(-1,1,13)
   >>> TW.refine([x_ffine]*d)

.. figure:: _static/Figures/TensorWrapperRefine.*

   The global tensor and the two views defined, after the grid refinement.

Quantics extension
^^^^^^^^^^^^^^^^^^

The quantics extension is used for extending the indices of the tesnor to the next power of ``Q``. The extension is performed so that the last coordinate point is appended to the coordinate points the necessary number of times. In order to apply the extension on a particular view, one needs to activate the view and then use the method :py:meth:`TensorWrapper.set_Q`.

.. doctest:: tw

   >>> TW.set_active_view('full')
   >>> TW.get_view_shape()
   (13, 13)
   >>> TW.get_extended_shape()
   (13, 13)
   >>> TW.set_Q(2)
   >>> TW.get_extended_shape()
   (16, 16)
   >>> TW.get_shape()
   (16, 16)
   >>> TW.shape
   (16, 16)

We can see that :py:meth:`TensorWrapper.get_extended_shape` returns the same output of :py:meth:`TensorWrapper.get_viw_shape` if no quantics extension has been applied.

Using the following code we can investigate the content of the extended tensor wrapper and plot it as shown in the following figure.

>>> A = TW[:,:]
>>> import matplotlib.pyplot as plt
>>> plt.figure(figsize=(6,5))
>>> plt.imshow(A,interpolation='none')
>>> plt.tight_layout()
>>> plt.show(False)

.. figure:: _static/Figures/TensorWrapperQExtension.*

   The *Quantics* extension applied to the ``full`` view results in the repetition of its limit values in the tensor grid.

Reshape
^^^^^^^

The shape of each view can be changed as long as the size returned by :py:meth:`TensorWrapper.get_extended_size` is unchanged. This means that if *no quantics* extension has been applied, the size must correspond to :py:meth:`TensorWrapper.get_view_size`. If a *quantics* extension has been applied, the size must correspond to :py:meth:`TensorWrapper.get_extended_size`.

For example let us reshape the *quantics* extended ``full`` view of the tensor to the shape (4,16).

.. doctest:: tw
   
   >>> TW.set_active_view('full')
   >>> TW.reshape((8,32))
   >>> TW.get_extended_shape()
   (16, 16)
   >>> TW.get_shape()
   (8, 32)
   >>> TW.shape
   (8, 32)

This results in the following reshaping of the tensor view:

>>> A = TW[:,:]
>>> import matplotlib.pyplot as plt
>>> plt.figure(figsize=(12,5))
>>> plt.imshow(A,interpolation='none')
>>> plt.tight_layout()
>>> plt.show(False)

.. figure:: _static/Figures/TensorWrapperReshape.*

   Reshaping of the *quantics* extended ``full`` view.

The *quantics* extension is used mainly to obtain a complete folding of base ``Q``. In this case this is obtained by:

.. doctest:: tw

   >>> import math
   >>> TW.reshape( [2] * int(round(math.log(TW.size,2))) )
   >>> TW.get_extended_shape()
   (16, 16)
   >>> TW.get_shape()
   (2, 2, 2, 2, 2, 2, 2, 2)
   >>> TW.shape
   (2, 2, 2, 2, 2, 2, 2, 2)

We finally can reset the shape to the *view* shape using:

.. doctest:: tw
   
   >>> TW.reset_shape()

Summary of shapes
^^^^^^^^^^^^^^^^^

Information regarding several shape transformations are always hold in the data structure. A hierarchy of shapes is used. The top shape is the **global** shape. In the following table we list the different shapes, their description and the main functions related and affecting them.

+-------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Shape**               | **Description**                                                                                                                                                                                                                       | **Functions**                                                                                                                                                                                                                                             |
+-------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Global                  | This is the underlying shape of the :py:class:`~TensorToolbox.TensorWrapper`.                                                                                                                                                         | :py:meth:`~TensorWrapper.get_global_shape`, :py:meth:`~TensorWrapper.get_global_ndim`, :py:meth:`~TensorWrapper.get_global_size`, :py:meth:`~TensorWrapper.refine`                                                                                        |
+-------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| View                    | Multiple views can be defined for a :py:class:`~TensorToolbox.TensorWrapper`. The views are defined as nested grids into the global grid. The default view is called ``full`` and is defined automatically at construction time       | :py:meth:`~TensorWrapper.set_view`, :py:meth:`~TensorWrapper.set_active_view`, :py:meth:`~TensorWrapper.get_view_shape`, :py:meth:`~TensorWrapper.get_view_ndim`, :py:meth:`~TensorWrapper.get_view_size`, :py:meth:`~TensorWrapper.refine`               |
+-------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Quantics Extended       | Each view can be extended to the next power of ``Q`` in order to allow the *quantics* folding :cite:`Khoromskij2011,Khoromskij2010` of the tensor.                                                                                    | :py:meth:`~TensorWrapper.set_Q`, :py:meth:`~TensorWrapper.get_extended_shape`, :py:meth:`~TensorWrapper.get_extended_ndim`, :py:meth:`~TensorWrapper.get_extended_size`                                                                                   |
+-------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Reshape                 | This is the result of the reshape of the tensor. If any of the preceding shape transformations have been applied, then the reshape is applied to the lowest transformation.                                                           | :py:meth:`~TensorWrapper.reshape`, :py:meth:`~TensorWrapper.get_shape`, :py:meth:`~TensorWrapper.get_ndim`, :py:meth:`~TensorWrapper.get_size`, :py:attr:`~TensorWrapper.shape`, :py:attr:`~TensorWrapper.ndim`, :py:attr:`~TensorWrapper.size`           |
+-------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. warning:: If a shape at any level is modified, every lower reshaping is automatically erased, due to possible inconsistency. For example, if a view is modified, any quantics extension and/or reshape of the view are reset.

.. note:: The :py:meth:`~TensorWrapper.refine` function erases all the quantics extensions and the reshapes of each view, but not the views themselves. Instead for each view, the :py:meth:`~TensorWrapper.refine` function updates the corresponding indices, fitting the old views to the new refinement.

Storage
^^^^^^^

Instances of the class :py:class:`~TensorToolbox.TensorWrapper` can be stored in files and reloaded as needed. The class :py:class:`~TensorToolbox.TensorWrapper` extends the class :py:class:`~TensorToolbox.storable_object`, which is responsible for storing objects in the :py:mod:`TensorToolbox`.

For the sake of efficiency and readability of the code, the :py:class:`~TensorToolbox.TensorWrapper` is stored in two different files with a common file name ``filename``:

* ``filename.pkl`` is a serialized version of the object thorugh the `pickle <https://docs.python.org/2/library/pickle.html>`_ library. The :py:class:`~TensorToolbox.TensorWrapper` serializes a minimal amount of auxiliary information needed for the definition of shapes, meshes, etc. The allocated data are not serialized using pickle, because when the amount of data is big, this would result in a very slow storage.
* ``filename.h5`` is a binary file containing the allocated data of the :py:class:`~TensorToolbox.TensorWrapper`. This file is generated using `h5py <http://www.h5py.org/>`_ and results in fast loading, writing and appending of data.

Let us store the :py:class:`~TensorToolbox.TensorWrapper`, we have been using up to now.

.. doctest:: tw

   >>> TW.set_store_location('tensorwrapper')
   >>> TW.store(force=True)

Check that the files have been stored:

.. code-block:: bash
   
   $ ls
   tensorwrapper.h5  tensorwrapper.pkl  WrapperExample.py

Let's now reload the :py:class:`~TensorToolbox.TensorWrapper`:

.. doctest:: tw

   >>> TW = TT.load('tensorwrapper')

The storage of the tensor wrapper can also be triggered using a timer. This is mostly useful when many time consuming computations need to be performed in order to allocate the desired entries of the tensor, and one wants to have always a backup copy of the data. The trigger for the storage is checked any time a new entry needs to be allocated fo storage.

For example, we can set the storage frequency to 5s:

.. doctest:: tw

   >>> import time
   >>> TW.data = {}
   >>> TW[1,2]
   >>> TW.set_store_freq( 5 )
   >>> time.sleep(6.0)
   >>> TW[3,5]

Checking the output we see:

.. code-block:: bash
   
   $ ls
   tensorwrapper.h5      tensorwrapper.pkl     WrapperExample.py
   tensorwrapper.h5.old  tensorwrapper.pkl.old

where the files ``.pkl`` and ``.h5`` are the files stored when the time-trigger is activated, while the files ``.pkl.old`` and ``h5.old`` are backup files containing the data stored in the previous example.
