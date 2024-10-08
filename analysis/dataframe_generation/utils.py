import os
import pathlib
import platform
import numpy as np
import slab
from dataclasses import dataclass

country_converter = {
    0: "Belgium",
    1: "Britain",
    2: "Congo",
    3: "Cuba",
    4: "Japan",
    5: "Mali",
    6: "Oman",
    7: "Peru",
    8: "Sudan",
    9: "Syria",
    10: "Togo",
    11: "Tonga",
    12: "Yemen",
}


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

def add_line_of_equality(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def add_grid_line_of_equality(g, task_name=None):
    task_name = task_name or g.data["task_name"].values[0]
    for ax in g.axes.flatten():
        ax_min = ay_min = 1.5 if "distance" in ax.get_title() else -55
        ax_max = ay_max = 12.5 if "distance" in ax.get_title() else 55
        if task_name == "nj":
            ax_min = ay_min = 1.5
            ax_max = ay_max = 6.5
        elif task_name == "su":
            ay_min = -15
            ay_max = 1
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ay_min, ay_max)
        add_line_of_equality(ax, color="grey", ls="--", alpha=.3)

@dataclass
class Speaker:
    """
    Class for handling the loudspeakers which are usually loaded using `read_speaker_table().`.
    """
    index: int  # the index number of the speaker
    analog_channel: int  # the number of the analog channel to which the speaker is attached
    analog_proc: str  # the processor to whose analog I/O the speaker is attached
    digital_proc: str  # the processor to whose digital I/O the speaker's LED is attached
    azimuth: float  # the azimuth angle of the speaker
    elevation: float  # the azimuth angle of the speaker
    digital_channel: int  # the int value of the bitmask for the digital channel to which the speakers LED is attached
    level: float = None  # the constant for level equalization
    filter: slab.Filter = None  # filter for equalizing the filters transfer function

    def __repr__(self):
        if (self.level is None) and (self.filter is None):
            calibrated = "NOT calibrated"
        else:
            calibrated = "calibrated"
        return f"<speaker {self.index} at azimuth {self.azimuth} and elevation {self.elevation}, {calibrated}>"


def read_speaker_table():
    """
    Read table containing loudspeaker information from a file and initialize a `Speaker` instance for each entry.

    Returns:
        (list): a list of instances of the `Speaker` class.
    """
    speakers = []
    table_file = pathlib.Path(os.getcwd()) / "analysis" / "dataframe_generation" / 'FREEFIELD_speakers.txt'
    table = np.loadtxt(table_file, skiprows=1, delimiter=",", dtype=str)
    for row in table:
        speakers.append(Speaker(index=int(row[0]), analog_channel=int(row[1]), analog_proc=row[2],
                                azimuth=float(row[3]), digital_channel=int(row[5]) if row[5] else None,
                                elevation=float(row[4]), digital_proc=row[6] if row[6] else None))
    return speakers

SPEAKERS = read_speaker_table()


def pick_speakers(picks):
    """
    Either return the speaker at given coordinates (azimuth, elevation) or the
    speaker with a specific index number.

    Args:
        picks (list of lists, list, int): index number of the speaker

    Returns:
        (list):
    """
    if isinstance(picks, (list, np.ndarray)):
        if all(isinstance(p, Speaker) for p in picks):
            speakers = picks
        elif all(isinstance(p, (int, np.int64, np.int32)) for p in picks):
            speakers = [s for s in SPEAKERS if s.index in picks]
        else:
            speakers = [s for s in SPEAKERS if (s.azimuth, s.elevation) in picks]
    elif isinstance(picks, (int, np.int64, np.int32)):
        speakers = [s for s in SPEAKERS if s.index == picks]
    elif isinstance(picks, Speaker):
        speakers = [picks]
    else:
        speakers = [s for s in SPEAKERS if (s.azimuth == picks[0] and s.elevation == picks[1])]
    if len(speakers) == 0:
        print("no speaker found that matches the criterion - returning empty list")
    return speakers

