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

    **DataSubType != Average** → Maps segement entity content of the HDF5 :ref:`mcs-hdf5-raw-segmentstream-label` to Python structures.
   
.. autoclass:: AverageSegmentTuple 
   :members:
   :member-order: bysource
   
.. autoclass:: AverageSegmentEntity 
   :members:
   :member-order: bysource
   
   **DataSubType == Average** → Maps segment entity content of the HDF5 :ref:`mcs-hdf5-raw-segmentstream-subtype-average-label` to Python structures.
   
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

The ''McsCMOSMEA'' module
-------------------------

.. module:: McsPy.McsCMOSMEA

.. autoclass:: McsCMOSMEAData
    :members:
    :member-order: bysource

.. autoclass:: McsGroup
    :members:
    :member-order: bysource

.. autoclass:: McsDataset
    :members:
    :member-order: bysource

*Raw Data (.cmcr)* files
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Acquisition
    :members:
    :member-order: bysource

.. autoclass:: McsInfo
    :members:
    :member-order: bysource

.. autoclass:: McsStream
    :members:
    :member-order: bysource

.. autoclass:: McsChannelStream
    :members:
    :member-order: bysource

.. autoclass:: McsChannelEntity
    :members:
    :member-order: bysource

.. autoclass:: McsEventStream
    :members:
    :member-order: bysource

.. autoclass:: McsEventEntity
    :members:
    :member-order: bysource

.. autoclass:: McsSensorStream
    :members:
    :member-order: bysource

.. autoclass:: McsSensorEntity
    :members:
    :member-order: bysource

.. autoclass:: McsSpikeStream
    :members:
    :member-order: bysource

.. autoclass:: McsSpikeEntity
    :members:
    :member-order: bysource

.. autoclass:: McsSegmentStream
    :members:
    :member-order: bysource

.. autoclass:: McsSegmentStreamEntity
    :members:
    :member-order: bysource

.. autoclass:: McsTimeStampStream
    :members:
    :member-order: bysource

.. autoclass:: McsTimeStampStreamEntity
    :members:
    :member-order: bysource

*Processed Data (.cmtr)* files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NetworkExplorer
    :members:
    :member-order: bysource

.. autoclass:: STAEntity
    :members:
    :member-order: bysource

.. autoclass:: SpikeExplorer
    :members:
    :member-order: bysource

.. autoclass:: SpikeSorter
    :members:
    :member-order: bysource

.. autoclass:: SpikeSorterUnitEntity
    :members:
    :member-order: bysource

.. autoclass:: FilterTool
    :members:
    :member-order: bysource

.. autoclass:: ActivitySummary
    :members:
    :member-order: bysource