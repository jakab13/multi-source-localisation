"""
system configurations used in speaker calibration
"""
IMAGING = {'speaker_file': 'IMAGING_speakers.txt',
           'device': {'file_name': 'TDT_RX6RX8_speaker_calibration.py',
                      'device_class': 'RX6RX8SpeakerCal'},
           'TDT_rec': 'RX6',
           'TDT_aud': 'RX8',
           'ref_spk_id': 0
           }

TESTING = {'speaker_file': 'TESTING_speakers.txt',
           'device': {},
           'TDT_rec': 'RX6',
           'TDT_aud': 'RX8',
           'ref_spk_id': 0
           }

FREEFIELD = {"speaker_file": "dome_speakers.txt",
             "device": {"file_name": "RX8.py",
                        "device_class": "RX8RX8SpeakerCal"},
             "TDT_rec": "RX8",
             "TDT_aud": "RX8",
             "ref_spk_id": 0
             }
