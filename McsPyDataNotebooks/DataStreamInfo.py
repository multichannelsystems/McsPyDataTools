#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import os
import McsPy
import McsPy.McsData
import argparse
import datetime
from tabulate import tabulate


# test_raw_data_file_path = r"TestData\mcd_Tutorial\1,2.h5"
# test_raw_data_file_path = r"TestData\mcd_Tutorial\Neuro_OTC_Spikes_Demo.h5"
# test_raw_data_file_path = r"TestData\mcd_Tutorial\PPF_Data.h5"
# test_raw_data_file_path = r"TestData\mcd_Tutorial\LinearLayout_StandardRecording.h5"


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
    print("")
    print(h5filename)
    print("")
    t_row = []
    d = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=int(raw_data.date_in_clr_ticks)/10)
    t_row.append(str(d.strftime("%Y-%m-%d %H:%M:%S")))
    t_row.append(raw_data.program_name)
    t_row.append(raw_data.program_version)
    real_row = []
    real_row.append(t_row)
    table_header = ["Date", "Program", "Version"]
    print(tabulate(real_row, headers=table_header))


def get_number_of_streams(rec, stream_type):
    num_streams = 0
    try:
        if stream_type == "analog":
            if rec.analog_streams != None:
                num_streams = len(rec.analog_streams)
        if stream_type == "event":
            if rec.event_streams != None:
                num_streams = len(rec.event_streams)
        if stream_type == "segment":
            if rec.segment_streams != None:
                num_streams = len(rec.segment_streams)
        if stream_type == "timestamp":
            if rec.timestamp_streams != None:
                num_streams = len(rec.timestamp_streams)
    except KeyError, e:
        print(e.message)
        num_streams = 0
    except ValueError, e:
        print(e.message)
        num_streams = 0
    return num_streams


def print_number_of_streams(rec, stream_type):
    num_streams = get_number_of_streams(rec, stream_type)
    print("number of {} streams: {}".format(stream_type, num_streams))


def get_streams_of_type(rec, stream_type):
    if stream_type == "analog":
        return rec.analog_streams
    elif stream_type == "event":
        return rec.event_streams
    elif stream_type == "segment":
        return rec.segment_streams
    elif stream_type == "timestamp":
        return rec.timestamp_streams
    else:
        assert False, "unknown stream type"
        return None


def get_info_rows(streams):
    rows = []
    x = streams.iteritems()
    for id, s in x:
        row = []
        row.append(s.stream_type)
        row.append(s.label)
        try:
            l = len(s.channel_infos)
            row.append(l)
        except AttributeError:
            row.append("")
        rows.append(row)
    return rows


def get_stream_info_rows(rec, stream_type):
    num_streams = get_number_of_streams(rec, stream_type)
    if num_streams > 0:
        rows = get_info_rows(get_streams_of_type(rec, stream_type))
        return rows


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


def print_file_info(h5filename):

    McsPy.McsData.VERBOSE = False

    raw_data = McsPy.McsData.RawData(h5filename)
    print_header_info(h5filename, raw_data)
    recording = raw_data.recordings[0]
    print_stream_info(recording)


def print_file_info2(h5filename):

    McsPy.McsData.VERBOSE = False
    raw_data = McsPy.McsData.RawData(h5filename)
    print_header_info(h5filename, raw_data)
    print("")

    recording = raw_data.recordings[0]
    all_rows = []
    for n in get_stream_type_names():
        if recording != None:
            rows = get_stream_info_rows(recording, n)
            if rows != None:
                for r in rows:
                    all_rows.append(r)

    table_header = ["Type", "Stream", "# ch"]
    print(tabulate(all_rows, headers=table_header))

#def tab_print():
#    print "\n".join (map (lambda (x, y): "%s\t%s" % ("\t".join (x), y), mylist) )

#def test():
#    raw_data = McsPy.McsData.RawData(test_raw_data_file_path)
#    print(raw_data.date)   


def parse_arguments():
    global parser
    parser = argparse.ArgumentParser(description="Get file and stream info from hdf5 files")
    parser.add_argument("-d", "--directory", help="directory where to look for hdf5 files")
    parser.add_argument("-f", "--file", help="filename")
    args = parser.parse_args()
    return args


def get_directory(args):
    return args.directory


def get_stream_type_names():
    stream_types = ["analog", "event", "segment", "timestamp"]
    return stream_types


def get_table_stream_info(recording):
    info = []
    for stream_type in get_stream_type_names():
        num_streams = get_number_of_streams(recording, stream_type)
        info.append(str(num_streams))
    return info


def get_table_row(f):
    try:
        raw_data = McsPy.McsData.RawData(f)
        row = []
        row.append(os.path.basename(f))
        if raw_data.recordings == None:
            row.append("Error: no raw data")
            # append empty columns
            for i in range(len(get_stream_type_names())):
                row.append("")
        else:
            d = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=int(raw_data.date_in_clr_ticks)/10)
            row.append(d.strftime("%Y-%m-%d %H:%M:%S"))
            recording = raw_data.recordings[0]
            l = get_table_stream_info(recording)
            for i in l:
                row.append(i)
        return row
    except IOError, e:
        print("IOError")
        print("Could not open " + f + "\n" + e.message)
        exit(1)

def print_dir_file_info(h5files):
    McsPy.McsData.VERBOSE = False

    table_header = ["File", "Date", "Anal.", "Ev", "Seg.", "TS"]
    # tf = TableFormatter(table_header)
    table = []
    for f in h5files:
        row = get_table_row(f)
        table.append(row)
    print(tabulate(table, headers=table_header))


def usage():
    parser.print_help()


def data_stream_info():
    args = parse_arguments()
    if args.directory != None:
        file_dir = get_directory(args)
    else:
        file_dir = ""
        if args.file == None:
            usage()
            exit()

    if args.file != None:
        filepath = os.path.join(file_dir, args.file)
        print_file_info2(filepath)
    elif file_dir != "":
        files = os.listdir(unicode(file_dir))
        only_files = [ f for f in files if os.path.isfile(os.path.join(unicode(file_dir), f)) ]

        if len(only_files) == 0:
            print("no files found in " + file_dir)
        else:
            only_files = [os.path.join(unicode(file_dir), f) for f in only_files]
            print_dir_file_info(only_files)

if __name__ == "__main__":
    data_stream_info()
