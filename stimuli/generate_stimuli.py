# Command line: tts help
# [-h] [--list_models [LIST_MODELS]] [--model_info_by_idx MODEL_INFO_BY_IDX] [--model_info_by_name MODEL_INFO_BY_NAME]
# [--text TEXT] [--model_name MODEL_NAME] [--vocoder_name VOCODER_NAME] [--config_path CONFIG_PATH]
# [--model_path MODEL_PATH] [--out_path OUT_PATH] [--use_cuda USE_CUDA] [--vocoder_path VOCODER_PATH]
# [--vocoder_config_path VOCODER_CONFIG_PATH] [--encoder_path ENCODER_PATH]
# [--encoder_config_path ENCODER_CONFIG_PATH] [--speakers_file_path SPEAKERS_FILE_PATH]
# [--language_ids_file_path LANGUAGE_IDS_FILE_PATH] [--speaker_idx SPEAKER_IDX] [--language_idx LANGUAGE_IDX]
# [--speaker_wav SPEAKER_WAV [SPEAKER_WAV ...]] [--gst_style GST_STYLE] [--capacitron_style_wav CAPACITRON_STYLE_WAV]
# [--capacitron_style_text CAPACITRON_STYLE_TEXT] [--list_speaker_idxs [LIST_SPEAKER_IDXS]]
# [--list_language_idxs [LIST_LANGUAGE_IDXS]] [--save_spectogram SAVE_SPECTOGRAM] [--reference_wav REFERENCE_WAV]
# [--reference_speaker_idx REFERENCE_SPEAKER_IDX] [--progress_bar PROGRESS_BAR]

# Import required libraries
import subprocess
import os
import pathlib
import ast
from stimuli.tts_models import models, get_from_c_arg
import slab
import random
import pyloudnorm as pyln
import numpy as np
import pandas as pd
import seaborn as sns
import datetime

DIR = pathlib.Path(os.getcwd())
save_directory = DIR / "samples" / "TTS" / "tts-countries_n13_resamp_48828"

tts_models = models["tts_models"]  # Between 1-58
vocoder_models = models["vocoder_models"]  # Between 1-16


def get_process_output(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    data, err = process.communicate()
    if process.returncode == 0:
        return data.decode('ascii')
    else:
        print("Error:", err)
    return ""

for tts_model_id, tts_model in tts_models.items():
    if "speaker_idxs" in tts_model:
        continue
    tts_c_arg = tts_model["c_arg"]
    language = get_from_c_arg(tts_c_arg, attr="language")
    if language == "en" or language == "multilingual":
        cmd = " ".join(["tts", "--model_name", tts_c_arg, "--list_speaker_idxs"])
        out_lines = get_process_output(cmd)
        out_lines = out_lines.decode('ascii')
        out_lines = out_lines.splitlines()
        try:
            speaker_idxs_obj = ast.literal_eval(out_lines[-1])
        except ValueError:
            print("last line is not a dict")
            continue
        tts_models[tts_model_id]["speker_idxs"] = speaker_idxs_obj


numbers = ["one", "two", "three", "four", "five", "six", "eight", "nine"]
numbers_numeric = ["1", "2", "3", "4", "5", "6", "8", "9"]
numbers_converter = dict(zip(numbers_numeric, numbers))
sentences = [
    "A king ruled the state in the early days.",
    "The ship was torn apart on the sharp reef.",
    "Sickness kept him home the third week.",
    "The wide road shimmered in the hot sun.",
    "The lazy cow lay in the cool grass.",
    "Lift the square stone over the fence.",
    "The rope will bind the seven books at once.",
    "Hop over the fence and plunge in.",
    "The friendly gang left the drug store.",
    "Mesh wire keeps chicks inside."
]
sentences = ["One and two and three and four and five and.",
             "Six and eight and nine and."]
countries = [
    "Belgium",
    "Britain",
    "Burma",
    "China",
    "Congo",
    "Cuba",
    "Haiti",
    "Japan",
    "Korea",
    "Libya",
    "Mali",
    "Mexico",
    "Nauru",
    "Norway",
    "Oman",
    "Peru",
    "Russia",
    "Sudan",
    "Syria",
    "Togo",
    "Tonga",
    "Turkey",
    "Yemen",
    "Zambia"
]

tts_model = tts_models[16]
tts_c_arg = tts_model["c_arg"]

eight = ["8"]
selected_speaker_ids = ['p248', 'p229', 'p318', 'p245', 'p256', 'p284', 'p307', 'p268']
selected_speaker_ids_male = ['p229', 'p318', 'p256', 'p307']
selected_speaker_ids_female = ['p248', 'p245', 'p284', 'p268']
all_speaker_ids = list(tts_model["speaker_idxs"].keys())

save_directory = DIR / "samples" / "TTS" / "tts-numbers_numeric_n13_resamp_48828"

for text in numbers_numeric:
    for speaker_id in selected_speaker_ids:
        sex = tts_model["speaker_genders"][speaker_id]
        filepath = save_directory.parent / str("talker-" + speaker_id + "_" +
                                        "sex-" + sex + "_" +
                                        "text-" + "_" + numbers_converter[text] + "_" + ".wav")
        args = [
            "tts",
            "--text", text,
            "--model_name", tts_c_arg,
            "--out_path", filepath,
            "--speaker_idx", speaker_id
            ]
        subprocess.run(args)

file_names = os.listdir(save_directory)
# i = 0
# for file_name in file_names:
#     old_file_path = save_directory / file_name
#     if "sex" not in file_name:
#         speaker_id = file_name[file_name.find("talker-") + len("talker-"):file_name.rfind('_text')]
#         sex = tts_model["speaker_genders"][speaker_id]
#         sex_string = "_sex-" + sex
#         new_file_name = file_name[:file_name.rfind('_text')] + sex_string + file_name[file_name.rfind('_text'):]
#         new_file_path = save_directory / new_file_name
#         print(old_file_path, new_file_path)
#         print(i)
#         os.rename(old_file_path, new_file_path)
#         i += 1

number_pool = ['one', 'two', 'three', 'four', 'five', 'six', 'eight', 'nine']
meter = pyln.Meter(22050, block_size=0.200)
for i in range(200):
    babble_ids_male = random.sample(selected_speaker_ids_male, 3)
    babble_ids_female = random.sample(selected_speaker_ids_female, 3)
    babble_ids = babble_ids_male + babble_ids_female
    babble_numbers = random.sample(number_pool, 6)
    sounds = []
    babble_sound = slab.Binaural.silence(duration=2.0, samplerate=22050)
    prev_max_length = 0
    max_length = 0
    for idx, babble_id in enumerate(babble_ids):
        file_name = [f for f in file_names if babble_id in f and babble_numbers[idx] in f][0]
        sound = slab.Binaural(save_directory / file_name)
        env = sound.envelope()
        sound.data = sound.data[::-1]
        sound_first_non_zero = next((i for i, x in enumerate(sound[:, 0]) if x), None)
        sound.data = sound.data[sound_first_non_zero:]
        shift = int(random.randint(0, sound.n_samples))
        sound.data = np.roll(sound.data, -shift)
        sound = sound.ramp(duration=0.05)
        max_length = max(prev_max_length, sound.n_samples)
        prev_max_length = max_length
        loudness = meter.integrated_loudness(sound.data)
        sound = slab.Binaural(pyln.normalize.loudness(sound.data, loudness, -25), samplerate=sound.samplerate)
        length_diff = babble_sound.n_samples - sound.n_samples
        silence = slab.Binaural.silence(duration=length_diff, samplerate=sound.samplerate)
        sound = slab.Binaural.sequence(sound, silence)
        babble_sound.data += sound.data
    first_non_zero = next((i for i, x in enumerate(babble_sound[:, 0]) if x), None)
    babble_sound.data = babble_sound.data[first_non_zero:max_length]
    babble_loudness = meter.integrated_loudness(babble_sound.data)
    babble_sound = slab.Binaural(pyln.normalize.loudness(babble_sound.data, babble_loudness, -25), samplerate=babble_sound.samplerate)
    babble_sound = babble_sound.ramp(duration=0.2, when="offset")
    babble_sound = babble_sound.ramp(duration=0.01, when="onset")
    # babble_sound.play()
    babble_file_name = "babble-reversed_n13_" + "-".join([item for t in list(zip(babble_ids, babble_numbers)) for item in t]) + ".wav"
    babble_sound.write(save_directory.parent / "babble-reversed-n13-shifted" / babble_file_name)

# Align reversed sounds
orig_DIR = DIR / "samples" / "TTS" / "tts-countries_n13_resamp_48828"
file_names = os.listdir(orig_DIR)
for file_name in file_names:
    if file_name == ".DS_Store":
        continue
    sound = slab.Binaural(orig_DIR / file_name)
    sound = sound.resize(duration=0.6)
    sound.data = sound.data[::-1]
    # sound_first_non_zero = next((i for i, x in enumerate(sound[:, 0]) if abs(x) > 0.03), None)
    # print(file_name, sound_first_non_zero)
    # sound.data = np.roll(sound.data, -sound_first_non_zero, axis=0)
    sound.write(DIR / "samples" / "TTS" / "tts-countries-reversed_n13_resamp_48828" / str(file_name[:-4] + "reversed.wav"), normalise=False)

# Spectral coverage
DIR = pathlib.Path(os.getcwd())
tts_models = models["tts_models"]  # Between 1-58
tts_model = tts_models[16]

talker_ids = ['p248', 'p229', 'p318', 'p245', 'p256', 'p284', 'p307', 'p268']
countries = ["Belgium", "Britain", "Congo", "Cuba", "Japan", "Mali", "Oman", "Peru", "Sudan", "Syria", "Togo", "Tonga", "Yemen"]
distances = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
trial_dur = 0.6
p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
upper_freq = 11000  # upper frequency limit that carries information for speech
dyn_range = 65
now = datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")

# setup = "cathedral"
for setup in ["cathedral"]:
    for stim_type in ["countries_forward", "countries_reversed"]:
        if setup == "cathedral":
            sub_DIR = DIR / "samples" / "TTS" / f"tts-{stim_type}_cathedral_n13_resamp_24414"
        else:
            if stim_type == "countries_forward":
                sub_DIR = DIR / "samples" / "TTS" / f"tts-countries_n13_resamp_24414"
            else:
                sub_DIR = DIR / "samples" / "TTS" / f"tts-countries-reversed_n13_resamp_24414"
        csv_outputpath = sub_DIR.parent / f"tts_spectral_coverage{now}.csv"
        for _ in range(10000):
            num_pres = random.choice([2, 3, 4, 5, 6])
            selected_talker_ids = random.sample(talker_ids, num_pres)
            selected_country_ids = random.sample(countries, num_pres)
            # selected_distances = random.sample(distances, num_pres)
            selected_distances = [7.0] * num_pres
            trial_composition = list()
            file_names = list()
            for n in range(num_pres):
                talker_id = selected_talker_ids[n]
                sex = tts_model["speaker_genders"][talker_id]
                country_id = selected_country_ids[n]
                distance = selected_distances[n]
                if setup == "cathedral":
                    country_id = country_id + "_reversed" if "reversed" in stim_type else country_id + "_"
                    file_name = f"sound-talker-{talker_id}_sex-{sex}_text-_{country_id}_mgb-level-27.5_distance-{distance}.wav"
                else:
                    if stim_type == "countries_forward":
                        file_name = f"talker-{talker_id}_sex-{sex}_text-_{country_id}_.wav"
                    else:
                        file_name = f"talker-{talker_id}_sex-{sex}_text-_{country_id}_reversed.wav"
                signal = slab.Sound(sub_DIR / file_name)
                trial_composition.append(signal.resize(trial_dur))
                file_names.append(file_name)
            sound = sum(trial_composition)
            sound = slab.Sound(sound.data.mean(axis=1), samplerate=sound.samplerate)
            sound = sound.resample(24414)
            freqs, times, power = sound.spectrogram(show=False)
            power = 10 * np.log10(power / (p_ref ** 2))  # logarithmic power for plotting
            power = power[freqs < upper_freq, :]
            dB_max = power.max()
            dB_min = dB_max - dyn_range
            interval = power[np.where((power > dB_min) & (power < dB_max))]
            percentage_filled = interval.shape[0] / power.flatten().shape[0]
            row = {
                "n_presented": num_pres,
                "talker_ids": selected_talker_ids,
                "country_ids": selected_country_ids,
                "distances": selected_distances,
                "speaker_ids": [d - 2 for d in selected_distances],
                "file_names": file_names,
                "spectral_coverage": percentage_filled,
                "stim_type": stim_type,
                "setup": setup
            }
            df_row = pd.DataFrame.from_dict([row])
            df_row.to_csv(csv_outputpath, mode='a', header=not os.path.exists(csv_outputpath), index=False)

df = pd.read_csv(csv_outputpath)
sns.displot(df, x="spectral_coverage", hue="n_presented", row="stim_type", col="setup", kind="kde", palette="winter")
