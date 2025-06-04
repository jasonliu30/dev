import BScan

######################
# Specify a Filename #
######################
### R-09        - 0.1-0.2 GB
input_scan = "scans/BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf"


#######################
# Load the BScan File #
#######################
'''
This function must be called first.
BScan.BScan(filename) loads the BScan file into memory and automatically reads the header information.
'''
b_scan = BScan.BScan(input_scan)


# Functions to access the different headers
'''
There are functions to get the header, extended header, and hardware information sections from b_scan.
'''
header = b_scan.get_header()
extended_header = b_scan.get_extended_header()
hw_info = b_scan.get_hardware_info()


##########################
# Get the List of Probes #
##########################
'''
b_scan.mapped_labels is a list of probes with the short-form labels (ie: NB1, NB2, CPC, APC)
The short-form labels is the label that all of the other functions expect
'''
probes = b_scan.mapped_labels
# Choose a probe
probe = probes[0]

'''
The Extended Header also contains a list of channel names, such as 'ID1 NB', 'Circ Shear P/C', or  'Axial Shear P/C'
However in D-Type scans, 'ID1 NB' contains 'NB1' and 'NB2' interleaved, which is why the
functions that get channel information expect the short-form labels.
'''
channel_labels = extended_header.ChannelLabels


#########################################
# Functions to Read Data from a Channel #
#########################################
'''
get_channel_info returns a class containing metadata about the specified channel.

This includes informatino such as:
  - axial, rotary, and time ranges
  - Labels (short & long)
  - Data Offset
  - Slice Offests (array of file offset for every slice)
  - Slice Sizes in Bytes

'''
channel = b_scan.get_channel_info(probe)


'''
get_channel_axes returns a class containing axis information of the secified channel.

This includes information such as:
  - axial position array
  - rotary position array
  - time position array
'''
axis = b_scan.get_channel_axes(probe)


'''
read_channel_data reads the UT Data for the specified channel. It returns a class containing the UT Data itself,
plus an axis class.

This includes information such as:
  - UT Data (3D Array of dimensions [axial range, rotary range, time range])
  - Axis class with the same information as above
'''
ut_scan_data = b_scan.read_channel_data(probe)

print("Complete")