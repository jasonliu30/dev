import pandas as pd
from collections import namedtuple
from typing import List

datamap_columns = pd.Index(['Indication', 'Axial Start', 'Rotary Start', 'Length (mm)',
                            'Width (deg)', 'Max Amp (dB)', 'Depth (mm)', 'Wall Thickness', 'Flaw',
                            'Flaw-group', 'Flaw-general', 'DR', 'Header', 'Date Produced',
                            'Station', 'Unit', 'Channel', 'Reactor Face', 'Channel End',
                            'Inspection Head', 'Operator Name', 'Collection Date',
                            'Collection Time', 'End of PT(1)', 'Roll 1(1)', 'Roll 2(1)',
                            'Roll 3(1)', 'Burnish(1)', 'Burnish(2)', 'Roll 3(2)', 'Roll 2(2)',
                            'Roll 1(2)', 'End of PT(2)', 'Outage Number', 'Filename', 'Axial_start',
                            'Axial_end', 'Rotary_start', 'Rotary_end', 'Filename end', 'Note1',
                            'Note2', 'Note3', 'Note 4', 'Note 5'],
                           dtype='object')


FlawLocation = namedtuple('FlawLocation', ['name', 'frame_start', 'frame_end', 'rotary_start', 'rotary_end', 'dr'])

