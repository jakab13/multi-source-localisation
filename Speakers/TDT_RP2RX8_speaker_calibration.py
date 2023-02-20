from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.config import get_config

from traits.api import CFloat, Str, CInt, Any, List
from labplatform.core import TDTblackbox as tdt
import numpy as np
import os
import time

import logging
log = logging.getLogger(__name__)


class RP2RX8SpeakerCalSetting(DeviceSetting):
    """
    Device class for speaker calibration
    """
    # 1 is the left entry channel
    rec_ch = CInt(1, group='primary', dsec='analog input channel to be used to record the sound',
                  reinit=False, tag_name='rec_ch', device='RP2')
    device_freq = CFloat(24414, group='status', dsec='sampling frequency of the device (Hz)')
    rcx_file_RP2 = Str('calibration\\RP2_rec_buf.rcx', group='status', dsec='the rcx file for RP2')
    rcx_file_RX8 = Str('calibration\\RX8_play_buf.rcx', group='status', dsec='the rcx file for RX8')
    processor_RP2 = Str('RP2', group='status', dsec='name of the processor')
    processor_RX8 = Str('RX8', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='')
    index = List([1, 2], group='primary', dsec='index of the device to connect to')
    max_stim_length_n = CInt(500000, group='status', dsec='maximum length for stimulus in number of data points')
    device_type = 'SpeakerCal_RP2RX8'


class RP2RX8SpeakerCal(Device):
    """
    the buffer 'PulseTTL' will not reset when calling pause/start. to reset the buffer, need to
    send software trigger 2 to the circuit, or use method reset_buffer
    """
    setting = RP2RX8SpeakerCalSetting()
    buffer = Any()
    RP2 = Any()
    RX8 = Any()
    _use_default_thread = False

    stimulus = Any()

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.RP2 = tdt.Processors()
        self.RP2 = tdt.initialize_processor(processor=self.setting.processor_RP2,
                                            connection=self.setting.connection,
                                            index=1,
                                            path=os.path.join(expdir, self.setting.rcx_file_RP2))
        self.RX8 = tdt.Processors()
        self.RX8.initialize(proc_list=[[f"{self.setting.processor_RX8}{self.setting.index[0]}",
                                           self.setting.processor_RX8,
                                        os.path.join(expdir, self.setting.rcx_file_RX8)],
                                        [f"{self.setting.processor_RX8}{self.setting.index[1]}",
                                        self.setting.processor_RX8,
                                        os.path.join(expdir, self.setting.rcx_file_RX8)]],
                            connection=self.setting.connection,
                            zbus=True)
        # not necessarily accurate
        TDT_freq = self.RP2.GetSFreq()  # not necessarily returns correct value
        if abs(TDT_freq - self.setting.device_freq) > 1:
            log.warning('TDT sampling frequency is different from that specified in software: {} vs. {}'.
                        format(TDT_freq, self.setting.device_freq))
            # self.setting.device_freq = self.handle.GetSFreq()
        self._output_specs = {'type': 'analog_signal',
                              'shape': (-1, 1),
                              'sampling_freq': self.setting.device_freq,
                              'dtype': np.float32,
                              }

    def _configure(self, **kwargs):
        pass

    def _pause(self):
        # ZBus trigger: (racknum, trigtype, delay (ms)); 0 = all racks; 0 = pulse, 1 = high, 2 = low; minimum of 2
        self.RX8.halt()

    def _start(self):
        self.RX8.trigger("zBusA", proc=self.RX8)
        self.wait_to_finish_playing()

    def _stop(self):
        self.RP2.halt()
        self.RX8.halt()

    def wait_to_finish_playing(self, proc="all", tag="playback"):
        if proc == "all":
            proc = list(self.RX8.procs.keys())
        elif isinstance(proc, str):
            proc = [proc]
        logging.info(f'Waiting for {tag} on {proc}.')
        while any(self.RX8.read(tag, proc=p) for p in proc):
            time.sleep(0.01)
        log.info('Done waiting.')

if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

