import bscan_structure_definitions as bscan
import os
import pandas as pd

from pathlib import Path

def format_2d_str_array(twoD_String_Array: [[str]]):
    twod_string = ''
    for row in twoD_String_Array:
        twod_string += ', '.join([str(thresh) for thresh in row]) + '\n'

    #strip the trailing newline to be compatible with other print statements that don't end with a newline
    return twod_string.rstrip('\n')


def save_bscan_data(b_scan_data: bscan.BScanData().data, save_dir: Path, channel_label: bscan.Probe, scan_count: int):
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    pd.DataFrame(b_scan_data).to_csv(os.path.join(save_dir / (channel_label.name + '_' + str(scan_count + 1) + '.csv')), header=False, index=False)


def save_footer_file(footer_info: bscan.ScanFooterData, file_location: str):

    formatted_footer_data = "ItemCount\n" + str(footer_info.item_count) + "\n\n"
    for footer_item in footer_info.footer_items:
        if isinstance(footer_item, bscan.FooterChangeTrackRecord):
            for change_track_record in footer_item.change_track_records:
                formatted_footer_data += "Change Track Record\n" + \
                            "AxialPosition\n" + str(change_track_record.axial_position) + "\n\n" + \
                            "Date and Time\n" + change_track_record.date_time + "\n\n" + \
                            "Parameter Name\n" + change_track_record.parameter_name + "\n\n" + \
                            "Parameter Value Before Change\n" + change_track_record.parameter_value_before_change + "\n\n" + \
                            "Parameter Value After Change\n" + change_track_record.parameter_value_after_change + "\n\n"

        elif isinstance(footer_item, bscan.FooterSoftwareGains):
            formatted_footer_data += "Software Gains\n" + ",".join([str(gain) for gain in footer_item.gains]) + "\n\n"

        elif isinstance(footer_item, bscan.FooterAnalystComments):
            formatted_footer_data += "Analyst Comments\n" + footer_item.comment_string + "\n\n"

    footer_file = open(os.path.join(file_location, 'Footer.csv'), 'w')
    footer_file.write(formatted_footer_data)
    footer_file.close()