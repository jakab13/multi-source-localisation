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

DIR = pathlib.Path(os.getcwd())
save_directory = DIR / "samples" / "TTS" / "numbers"

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


numbers = ["six", "seven", "eight", "nine"]
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

eight = ["three"]
selected_speaker_ids = ['p243', 'p229', 'p318', 'p245', 'p256', 'p284', 'p307', 'p280']
selected_speaker_ids_male = ['p229', 'p318', 'p256', 'p307']
selected_speaker_ids_female = ['p243', 'p245', 'p284', 'p280']
all_speaker_ids = list(tts_model["speaker_idxs"].keys())

for text in eight:
    for speaker_id in ['p307']:
        sex = tts_model["speaker_genders"][speaker_id]
        filepath = save_directory.parent / str("talker-" + speaker_id + "_" +
                                        "sex-" + sex + "_" +
                                        "text-" + "'" + text + "'" + ".wav")
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

number_pool = ['one', 'two', 'three', 'four', 'five']
meter = pyln.Meter(22050, block_size=0.200)
for i in range(200):
    babble_ids_male = random.sample(selected_speaker_ids_male, 2)
    babble_ids_female = random.sample(selected_speaker_ids_female, 2)
    babble_ids = babble_ids_male + babble_ids_female
    babble_numbers = random.sample(number_pool, 4)
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
    babble_file_name = "babble-reversed_" + "-".join([item for t in list(zip(babble_ids, babble_numbers)) for item in t]) + ".wav"
    babble_sound.write(save_directory.parent / "babble-reversed-shifted" / babble_file_name)