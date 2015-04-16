#!/usr/bin/env/python

import os
import McsPy
import McsPy.McsData
import argparse
#from TableFormatter import TableFormatter
from tabulate import tabulate


test_raw_data_file_path = r"TestData\mcd_Tutorial\1,2.h5"
test_raw_data_file_path = r"TestData\mcd_Tutorial\Neuro_OTC_Spikes_Demo.h5"    
test_raw_data_file_path = r"TestData\mcd_Tutorial\PPF_Data.h5"    
test_raw_data_file_path = r"TestData\mcd_Tutorial\LinearLayout_StandardRecording.h5"


def print_analog_channel_info(streams):
    x = streams.iteritems()
    for id, s in x:
        print(s.label)
        print("channels: {}".format(len(s.channel_infos)))


def print_event_channel_info(streams):
    x = streams.iteritems()
    for id, s in x:
        print(s.label)


def print_segment_channel_info(streams):
    x = streams.iteritems()
    for id, s in x:
        print(s.label)


def print_timestamp_channel_info(streams):
    x = streams.iteritems()
    for id, s in x:
        print(s.label)


def print_header_info(h5filename, raw_data):
    print("filename: " + h5filename)
    print("date: {0}".format(raw_data.date))
    print("program name: " + raw_data.program_name)
    print("program version: " + raw_data.program_version) 


def get_number_of_streams(rec, stream_type):
    num_streams = 0
    try:
        if stream_type == "analog":
            num_streams = len(rec.analog_streams)
        if stream_type == "event":
            num_streams = len(rec.event_streams)
        if stream_type == "segment":
            num_streams = len(rec.segment_streams)
        if stream_type == "timestamp":
            num_streams = len(rec.timestamp_streams)
    except KeyError:
        num_streams = 0
    return num_streams


def print_number_of_streams(rec, stream_type):
    num_streams = get_number_of_streams(rec, stream_type)
    print("number of {} streams: {}".format(stream_type, num_streams))


def print_stream_info(rec):
    #try:
    #    num_streams = len(rec.analog_streams)
    #except KeyError:
    #    num_streams = 0
    #print("number of analog streams: {}".format(num_streams))
    stream_type = "analog"
    num_streams = get_number_of_streams(rec, stream_type)
    print_number_of_streams(rec, stream_type)
    if num_streams > 0:
        print_analog_channel_info(rec.analog_streams)


    stream_type = "event"
    num_streams = get_number_of_streams(rec, stream_type)
    print_number_of_streams(rec, stream_type)
    if num_streams > 0:
        print_event_channel_info(rec.event_streams)

    stream_type = "segment"
    num_streams = get_number_of_streams(rec, stream_type)
    print_number_of_streams(rec, stream_type)
    if num_streams > 0:
        print_segment_channel_info(rec.event_streams)

    stream_type = "timestamp"
    num_streams = get_number_of_streams(rec, stream_type)
    print_number_of_streams(rec, stream_type)
    if num_streams > 0:
        print_timestamp_channel_info(rec.event_streams)

    #try:
    #    num_streams = len(rec.event_streams)
    #except KeyError:
    #    num_streams = 0
    #print("number of event streams: {}".format(num_streams))
    #if num_streams > 0:
    #    print_event_channel_info(rec.event_streams)


def print_file_info(h5filename):

    McsPy.McsData.verbose = False

    raw_data = McsPy.McsData.RawData(h5filename)
    print_header_info(h5filename, raw_data)

    recording = raw_data.recordings[0]

    print_stream_info(recording)

#    print("number of analog streams: {}".format(len(raw_data.recordings[0].analog_streams)))
    #num_streams = len(raw_data.recordings[0].analog_streams)

    #print(num_streams)

    #print_analog_stream_info(raw_data.recordings[0].analog_streams)

    #print("number of event streams")
    #try:
    #    num_streams = len(raw_data.recordings[0].event_streams)
    #    print(num_streams)
    #except KeyError:
    #    print("no event stream found")

    #for s in raw_data.recordings[0].analog_streams

#def tab_print():
#    print "\n".join (map (lambda (x, y): "%s\t%s" % ("\t".join (x), y), mylist) )

#def test():
#    raw_data = McsPy.McsData.RawData(test_raw_data_file_path)
#    print(raw_data.date)   

def parse_arguments():
    parser = argparse.ArgumentParser(description="Get file and stream info from hdf5 files")
    parser.add_argument("-d", "--directory", help="directory where to look for hdf5 files")
    parser.add_argument("-f", "--file", help="filename")
    args = parser.parse_args()
    return args


def get_directory(args):
    return args.directory


def get_table_stream_info(recording):
    info = []
    stream_types = ["analog", "event", "segment", "timestamp"]
    for stream_type in stream_types:
        num_streams = get_number_of_streams(recording, stream_type)
        info.append(str(num_streams))
    return info

def get_table_row(f):
    raw_data = McsPy.McsData.RawData(f)
    row = []
    row.append(os.path.basename(f))
    row.append("{0}".format(raw_data.date))
    recording = raw_data.recordings[0]
    l = get_table_stream_info(recording)
    for i in l:
        row.append(i)
    return row


def print_dir_file_info(h5files):
    McsPy.McsData.verbose = False

    table_header = ["File", "Date", "Anal.", "Ev", "Seg.", "TS"]
    # tf = TableFormatter(table_header)
    table = []
    for f in h5files:
        row = get_table_row(f)
        table.append(row)
    #     tf.addDataColumns(row)
    # print(tf.createTable())
    print(tabulate(table, headers=table_header))


def data_stream_info():
    args = parse_arguments()
    if args.directory != None:
        file_dir = get_directory(args)
    else:
        file_dir = ""

    if args.file != None:
        filepath = os.path.join(file_dir, args.file)
        #print("printing file info")
        print_file_info(filepath)
    elif file_dir != "":
        files = os.listdir(unicode(file_dir))
        only_files = [ f for f in files if os.path.isfile(os.path.join(unicode(file_dir), f)) ]

        if len(only_files) == 0:
            print("no files found in " + file_dir)
        else:
            # print("available files")
            # print(onlyfiles)
            # print(os.path.join(file_dir, onlyfiles[0]))

            only_files = [os.path.join(unicode(file_dir), f) for f in only_files]
            print_dir_file_info(only_files)

            # for f in only_files:
            #     print_file_info(f)
            #     #print_dir_file_info(os.path.join(file_dir, f))

if __name__ == "__main__":
    data_stream_info()
    #print_file_info(test_raw_data_file_path)
