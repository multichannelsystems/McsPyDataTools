import McsData

raw_data_file_path = "d:\\Programming\\MCSuite\\McsDataTools\\McsDataFileConverter\\bin\\Debug\\Experiment.h5"

raw_data = McsData.RawData(raw_data_file_path)
print(raw_data.comment)
print(raw_data.date)
print(raw_data.clr_date)
print(raw_data.date_in_clr_ticks)
print(raw_data.file_guid)
print(raw_data.mea_id)
print(raw_data.mea_name)
print(raw_data.program_name)
print(raw_data.program_version) 
print(raw_data.recordings)
print(raw_data.recordings[0].analog_streams)
