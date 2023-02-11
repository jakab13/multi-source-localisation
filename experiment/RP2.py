from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt
import time
from traits.api import CFloat, Str, Any, Tuple, Int, Float
import os
import logging
import numpy as np

log = logging.getLogger(__name__)


class RP2Setting(DeviceSetting):  # this class contains important settings for the device and sits in self.setting
    sampling_freq = CFloat(48288.125, group='primary', dsec='Sampling frequency of the device (Hz)', reinit=False)
    # buffer_size_max = CInt(50000, group='primary', dsec='Max Buffer size', reinit=False)
    file = Str('MSL\\RCX\\button_rec.rcx', group='status', dsec="Name of the rcx file to load")
    processor = Str('RP2', group='status', dsec='Name of the processor')
    connection = Str('GB', group='status', dsec='Connection of the device')
    index = Any(1, group='status', dsec='index of the device to connect to')
    device_name = Str("RP2", group="status", dsec="Name of the device")
    device_type = Str("Processor", group='status', dsec='type of the device')
    shape = Tuple(1, group="status", dsec="Dimension of the device output")
    dtype = Int(int, group="status", dsec="data type of the output")
    type = Str("analog_signal", group="status", dsec="Type of the signal")
    control_interval = Float(0.1, group="primary", dsec="Interval at which the device is checking its state")


class RP2Device(Device):

    setting = RP2Setting()
    handle = Any()  # handle for TDT method execution like handle.write, handle.read, ...
    # thread = Instance(threading.Thread)  # important for threading
    _output_specs = {'type': setting.type, 'sampling_freq': setting.sampling_freq,
                     'dtype': setting.dtype, "shape": setting.shape}
    _use_default_thread = True
    button_press_count = Int(0)


    def _initialize(self, **kwargs):  # this method is called upon self.initialize() execution
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor=self.setting.processor, connection=self.setting.connection,
                                               index=self.setting.index, path=os.path.join(expdir, self.setting.file))

    def _configure(self, **kwargs):  # device needs to be configured before each trial. Sets state to "ready".
        pass

    def _start(self):  # starts device
        pass

    def _pause(self):  # needs to be called after each trial
        pass

    def _stop(self):  # stops the device. Initialization necessary when this method is called
        self.handle.Halt()

    def wait_for_button(self):  # stops the circuit as long as no button is being pressed
        log.info("Waiting for button press ...")
        while not self.handle.GetTagVal("response"):
            time.sleep(0.1)  # sleeps while the response tag in the rcx circuit does not yield 1
        self.button_press_count += 1
        print(self.button_press_count)

    def get_response(self):  # collects response, preferably called right after wait_for_button
        log.info("Acquiring button response ... ")
        # because the response is stored in bit value, we need the base 2 log
        self._output_specs["response"] = int(np.log2(self.handle.GetTagVal("response")))
        return self._output_specs["response"]

    def thread_func(self):
        if self.experiment:
            if self.experiment().setting.experiment_name == "SpatMask" or "NumJudge":
                if self.handle.GetTagVal("response") > 0:
                    self.experiment().process_event({'trial_stop': 0})
            elif self.experiment().setting.experiment_name == "LocaAccu":
                if self.button_press_count % 2 != 0:
                    if self.button_press_count != 1:
                        self.experiment().process_event({'trial_stop': 0})
                    else:
                        pass


if __name__ == "__main__":
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    RP2 = RP2Device()

    log.addHandler(ch)
    responses = list()

    for trial in range(9):
        # RP2.configure()
        RP2.start()
        RP2.wait_for_button()
        responses.append(RP2.get_response())
        RP2.pause()
        print(RP2._output_specs["response"])
        time.sleep(1)
