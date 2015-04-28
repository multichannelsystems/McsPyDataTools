McsPyDataTools API Reference
============================

.. automodule:: McsPy
   :members:

The ''McsData'' module
----------------------

.. module:: McsPy.McsData

.. autoclass:: RawData  
   :members:
   :member-order: bysource

.. autoclass:: Recording  
   :members:
   :member-order: bysource
   
   Provides the content of the HDF5 :ref:`mcs-hdf5-raw-recording-label` in Python.
   
*Data-Stream*-Structures containing the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
.. autoclass:: Stream
   :members:
   :member-order: bysource
   
.. autoclass:: AnalogStream 
   :members:
   :member-order: bysource
   
   Provides the content of the HDF5 :ref:`mcs-hdf5-raw-analogstream-label` in Python.

.. autoclass:: FrameStream 
   :members:
   :member-order: bysource

   Provides the content of the HDF5 :ref:`mcs-hdf5-raw-framestream-label` in Python.
   
.. autoclass:: FrameEntity 
   :members:
   :member-order: bysource
   
   Provides the content of the HDF5 :ref:`mcs-hdf5-raw-framestream-label`
   and :ref:`mcs-hdf5-raw-framestream-entity-label` in Python.
   
.. autoclass:: EventStream 
   :members:
   :member-order: bysource

   Provides the content of the HDF5 :ref:`mcs-hdf5-raw-eventstream-label` in Python.
   
.. autoclass:: EventEntity 
   :members:
   :member-order: bysource
   
   Maps data event entity content of the HDF5 :ref:`mcs-hdf5-raw-eventstream-label` to Python structures.
   
.. autoclass:: SegmentStream 
   :members:
   :member-order: bysource

   Provides the content of the HDF5 :ref:`mcs-hdf5-raw-segmentstream-label` in Python.
   
.. autoclass:: SegmentEntity 
   :members:
   :member-order: bysource

    **DataSybType != Average** → Maps segement entity content of the HDF5 :ref:`mcs-hdf5-raw-segmentstream-label` to Python structures.
   
.. autoclass:: AverageSegmentTuple 
   :members:
   :member-order: bysource
   
.. autoclass:: AverageSegmentEntity 
   :members:
   :member-order: bysource
   
   **DataSybType == Average** → Maps segment entity content of the HDF5 :ref:`mcs-hdf5-raw-segmentstream-subtype-average-label` to Python structures.
   
.. autoclass:: TimeStampStream 
   :members:
   :member-order: bysource

   Provides the content of the HDF5 :ref:`mcs-hdf5-raw-timestampstream-label` in Python.
   
.. autoclass:: TimeStampEntity 
   :members:
   :member-order: bysource
   
   Maps data timestamp entity data of the HDF5 :ref:`mcs-hdf5-raw-timestampstream-label` to Python structures.
   
*Info*-Classes containing Meta-Information for the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Info
   :members:
   :member-order: bysource
   
.. autoclass:: ChannelInfo 
   :members:
   :member-order: bysource
 
.. autoclass:: InfoSampledData 
   :members:
   :member-order: bysource

.. autoclass:: EventEntityInfo
   :members:
   :member-order: bysource
   
.. autoclass:: SegmentEntityInfo
   :members:
   :member-order: bysource
  
.. autoclass:: TimeStampEntityInfo
   :members:
   :member-order: bysource

The ''McsCMOS'' module
----------------------

.. automodule:: McsPy.McsCMOS
   :members:

