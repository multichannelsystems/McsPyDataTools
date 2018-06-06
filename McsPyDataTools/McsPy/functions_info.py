# -*- coding: utf-8 -*-
#
#_________________________________________________________________
#
# (c) 2018 by Multi Channel Systems MCS GmbH
# All rights reserved
#    
#_________________________________________________________________

"""Functions to read and show infos of all streams"""
__docformat__ = 'restructuredtext'
__all__ = ['print_header_info']

##-- Imports

import os
import McsPy
import McsPy.McsData
#import argparse
import datetime
from tabulate import tabulate
from McsPy.McsData import *

##--- Functions

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

def print_stream_infos(streams, info_select_func):
    stream_items = streams.items()
    for id,s in stream_items:
        print(info_select_func(s))

def get_number_of_streams(rec, stream_type):
    num_streams = 0
    try:
        streams = get_streams_of_type(rec, stream_type)
        if streams != None:
            num_streams = len(streams)
    except KeyError as e:
        print(e)
        num_streams = 0
    except ValueError as e:
        print(e)
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
    elif stream_type == "frame":
        return rec.frame_streams
    else:
        assert False, "unknown stream type"
        return None

_num_channel_func_per_stream = {
    Stream.Stream_Types[0]: lambda analog_stream: len(analog_stream.channel_infos),
    Stream.Stream_Types[1]: lambda event_stream: len(event_stream.event_entity),
    Stream.Stream_Types[2]: lambda segment_stream: len(segment_stream.segment_entity),
    Stream.Stream_Types[3]: lambda timestamp_stream: len(timestamp_stream.timestamp_entity),
    Stream.Stream_Types[4]: lambda frame_stream: len(frame_stream.frame_entity)}

def get_info_rows(streams, channel_num_func):
    rows = []
    x = streams.items()
    for id, s in x:
        row = []
        row.append(s.stream_type)
        row.append(s.label)
        try:
            l = channel_num_func(s)
            row.append(l)
        except AttributeError:
            row.append("")
        rows.append(row)
    return rows

def get_stream_info_rows(rec, stream_type):
    num_streams = get_number_of_streams(rec, stream_type)
    if num_streams > 0:
        rows = get_info_rows(get_streams_of_type(rec, stream_type), _num_channel_func_per_stream[stream_type])
        return rows

def print_all_stream_infos(rec):
    stream_type = "analog"
    num_streams = get_number_of_streams(rec, stream_type)
    print_number_of_streams(rec, stream_type)
    if num_streams > 0:
        print_stream_infos(rec.analog_streams, lambda stream: "channels: {}".format(len(stream.channel_infos)))

    stream_type = "event"
    num_streams = get_number_of_streams(rec, stream_type)
    print_number_of_streams(rec, stream_type)
    if num_streams > 0:
        print_stream_infos(rec.event_streams, lambda stream: stream.label)

    stream_type = "segment"
    num_streams = get_number_of_streams(rec, stream_type)
    print_number_of_streams(rec, stream_type)
    if num_streams > 0:
        print_stream_infos(rec.segment_streams, lambda stream: stream.label)

    stream_type = "timestamp"
    num_streams = get_number_of_streams(rec, stream_type)
    print_number_of_streams(rec, stream_type)
    if num_streams > 0:
        print_stream_infos(rec.timestamp_streams, lambda stream: stream.label)

def print_file_info_short(h5filename):
    McsPy.McsData.VERBOSE = False
    raw_data = McsPy.McsData.RawData(h5filename)
    print_header_info(h5filename, raw_data)
    recording = raw_data.recordings[0]
    print_all_stream_infos(recording)

def print_file_info(h5filename):
    cache_verbosity = McsPy.McsData.VERBOSE
    McsPy.McsData.VERBOSE = False
    raw_data = McsPy.McsData.RawData(h5filename)
    print_header_info(h5filename, raw_data)
    print("")
    recording = raw_data.recordings[0]
    all_rows = []
    for n in Stream.Stream_Types:
        if recording != None:
            rows = get_stream_info_rows(recording, n)
            if rows != None:
                for r in rows:
                    all_rows.append(r)

    table_header = ["Type", "Stream", "# ch"]
    print(tabulate(all_rows, headers=table_header))
    McsPy.McsData.VERBOSE = cache_verbosity

def get_table_stream_info(recording):
    info = []
    for stream_type in Stream.Stream_Types:
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
            for i in range(len(Stream.Stream_Types)):
                row.append("")
        else:
            d = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=int(raw_data.date_in_clr_ticks)/10)
            row.append(d.strftime("%Y-%m-%d %H:%M:%S"))
            recording = raw_data.recordings[0]
            l = get_table_stream_info(recording)
            for i in l:
                row.append(i)
        return row
    except IOError as e:
        print("IOError")
        print("Could not open " + f + "\n" + e)
        exit(1)

def print_short_file_infos(h5files):
    McsPy.McsData.VERBOSE = False

    table_header = ["File", "Date", "Anal.", "Ev", "Seg.", "TS", "FS"]
    # tf = TableFormatter(table_header)
    table = []
    for f in h5files:
        row = get_table_row(f)
        table.append(row)
    print(tabulate(table, headers=table_header))

def print_dir_file_info(file_dir):
    if (file_dir != None) and (file_dir != ""):
        if os.path.isdir(file_dir):
            files = os.listdir(str(file_dir))
            only_files = [ f for f in files if os.path.isfile(os.path.join(str(file_dir), f))]
            only_h5_files = [f for f in only_files if os.path.splitext(f)[1] == '.h5']
            if len(only_h5_files) == 0:
                    print("no files found in " + file_dir)
            else:
                only_files = [os.path.join(str(file_dir), f) for f in only_h5_files]
                print_short_file_infos(only_files)
                #print_short_file_infos(['.\\McsPy\\tests\\TestData\\20150402_00 Atrium_002.h5'])
