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

DIR = pathlib.Path(os.getcwd())
save_directory = DIR / "samples" / "TTS"

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


numbers = ["one", "two", "three", "four", "five"]
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
tts_model = tts_models[16]
tts_c_arg = tts_model["c_arg"]
for text in sentences:
    for speaker_id in list(tts_model["speaker_idxs"].keys())[:10]:
        filepath = save_directory / str("talker-" + speaker_id + "_" + "text-" + "'" + text + "'" + ".wav")
        args = [
            "tts",
            "--text", text,
            "--model_name", tts_c_arg,
            "--out_path", filepath,
            "--speaker_idx", speaker_id
            ]
        subprocess.run(args)
