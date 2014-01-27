import McsData

raw_data_file_path = "d:\\Programming\\MCSuite\\McsDataTools\\McsDataFileConverter\\bin\\Debug\\Experiment.h5"

raw_data = McsData.RawData(raw_data_file_path)
#grp_attrs = raw_data.h5_file['Data'].attrs.iteritems()
#for (name, value) in grp_attrs: print(name, value)
session_info = raw_data.session_info
print(session_info)
