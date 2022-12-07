models = {
    "tts_models": {
        1: {
            "c_arg": "tts_models/multilingual/multi-dataset/your_tts",
            "speaker_idxs": {
                'female-en-5': 0,
                'female-en-5': 1,
                'female-pt-4': 2,
                'male-en-2': 3,
                'male-en-2': 4,
                'male-pt-3': 5
            }
        },
        2: {"c_arg": "tts_models/bg/cv/vits"},
        3: {"c_arg": "tts_models/cs/cv/vits"},
        4: {"c_arg": "tts_models/da/cv/vits"},
        5: {"c_arg": "tts_models/et/cv/vits"},
        6: {"c_arg": "tts_models/ga/cv/vits"},
        7: {"c_arg": "tts_models/en/ek1/tacotron2"},
        8: {"c_arg": "tts_models/en/ljspeech/tacotron2-DDC"},
        9: {"c_arg": "tts_models/en/ljspeech/tacotron2-DDC_ph"},
        10: {"c_arg": "tts_models/en/ljspeech/glow-tts"},
        11: {"c_arg": "tts_models/en/ljspeech/speedy-speech"},
        12: {"c_arg": "tts_models/en/ljspeech/tacotron2-DCA"},
        13: {"c_arg": "tts_models/en/ljspeech/vits"},
        14: {"c_arg": "tts_models/en/ljspeech/vits--neon"},
        15: {"c_arg": "tts_models/en/ljspeech/fast_pitch"},
        16: {"c_arg": "tts_models/en/vctk/vits",
             "speaker_idxs": {
                'p225': 1, 'p226': 2, 'p227': 3, 'p228': 4, 'p229': 5, 'p230': 6, 'p231': 7, 'p232': 8, 'p233': 9, 'p234': 10, 'p236': 11, 'p237': 12, 'p238': 13, 'p239': 14, 'p240': 15, 'p241': 16, 'p243': 17, 'p244': 18, 'p245': 19, 'p246': 20, 'p247': 21, 'p248': 22, 'p249': 23, 'p250': 24, 'p251': 25, 'p252': 26, 'p253': 27, 'p254': 28, 'p255': 29, 'p256': 30, 'p257': 31, 'p258': 32, 'p259': 33, 'p260': 34, 'p261': 35, 'p262': 36, 'p263': 37, 'p264': 38, 'p265': 39, 'p266': 40, 'p267': 41, 'p268': 42, 'p269': 43, 'p270': 44, 'p271': 45, 'p272': 46, 'p273': 47, 'p274': 48, 'p275': 49, 'p276': 50, 'p277': 51, 'p278': 52, 'p279': 53, 'p280': 54, 'p281': 55, 'p282': 56, 'p283': 57, 'p284': 58, 'p285': 59, 'p286': 60, 'p287': 61, 'p288': 62, 'p292': 63, 'p293': 64, 'p294': 65, 'p295': 66, 'p297': 67, 'p298': 68, 'p299': 69, 'p300': 70, 'p301': 71, 'p302': 72, 'p303': 73, 'p304': 74, 'p305': 75, 'p306': 76, 'p307': 77, 'p308': 78, 'p310': 79, 'p311': 80, 'p312': 81, 'p313': 82, 'p314': 83, 'p316': 84, 'p317': 85, 'p318': 86, 'p323': 87, 'p326': 88, 'p329': 89, 'p330': 90, 'p333': 91, 'p334': 92, 'p335': 93, 'p336': 94, 'p339': 95, 'p340': 96, 'p341': 97, 'p343': 98, 'p345': 99, 'p347': 100, 'p351': 101, 'p360': 102, 'p361': 103, 'p362': 104, 'p363': 105, 'p364': 106, 'p374': 107, 'p376': 108
             }},
        17: {"c_arg": "tts_models/en/vctk/fast_pitch",
             "speaker_idxs": {
                'VCTK_p225': 0, 'VCTK_p226': 1, 'VCTK_p227': 2, 'VCTK_p228': 3, 'VCTK_p229': 4, 'VCTK_p230': 5, 'VCTK_p231': 6, 'VCTK_p232': 7, 'VCTK_p233': 8, 'VCTK_p234': 9, 'VCTK_p236': 10, 'VCTK_p237': 11, 'VCTK_p238': 12, 'VCTK_p239': 13, 'VCTK_p240': 14, 'VCTK_p241': 15, 'VCTK_p243': 16, 'VCTK_p244': 17, 'VCTK_p245': 18, 'VCTK_p246': 19, 'VCTK_p247': 20, 'VCTK_p248': 21, 'VCTK_p249': 22, 'VCTK_p250': 23, 'VCTK_p251': 24, 'VCTK_p252': 25, 'VCTK_p253': 26, 'VCTK_p254': 27, 'VCTK_p255': 28, 'VCTK_p256': 29, 'VCTK_p257': 30, 'VCTK_p258': 31, 'VCTK_p259': 32, 'VCTK_p260': 33, 'VCTK_p261': 34, 'VCTK_p262': 35, 'VCTK_p263': 36, 'VCTK_p264': 37, 'VCTK_p265': 38, 'VCTK_p266': 39, 'VCTK_p267': 40, 'VCTK_p268': 41, 'VCTK_p269': 42, 'VCTK_p270': 43, 'VCTK_p271': 44, 'VCTK_p272': 45, 'VCTK_p273': 46, 'VCTK_p274': 47, 'VCTK_p275': 48, 'VCTK_p276': 49, 'VCTK_p277': 50, 'VCTK_p278': 51, 'VCTK_p279': 52, 'VCTK_p280': 53, 'VCTK_p281': 54, 'VCTK_p282': 55, 'VCTK_p283': 56, 'VCTK_p284': 57, 'VCTK_p285': 58, 'VCTK_p286': 59, 'VCTK_p287': 60, 'VCTK_p288': 61, 'VCTK_p292': 62, 'VCTK_p293': 63, 'VCTK_p294': 64, 'VCTK_p295': 65, 'VCTK_p297': 66, 'VCTK_p298': 67, 'VCTK_p299': 68, 'VCTK_p300': 69, 'VCTK_p301': 70, 'VCTK_p302': 71, 'VCTK_p303': 72, 'VCTK_p304': 73, 'VCTK_p305': 74, 'VCTK_p306': 75, 'VCTK_p307': 76, 'VCTK_p308': 77, 'VCTK_p310': 78, 'VCTK_p311': 79, 'VCTK_p312': 80, 'VCTK_p313': 81, 'VCTK_p314': 82, 'VCTK_p316': 83, 'VCTK_p317': 84, 'VCTK_p318': 85, 'VCTK_p323': 86, 'VCTK_p326': 87, 'VCTK_p329': 88, 'VCTK_p330': 89, 'VCTK_p333': 90, 'VCTK_p334': 91, 'VCTK_p335': 92, 'VCTK_p336': 93, 'VCTK_p339': 94, 'VCTK_p340': 95, 'VCTK_p341': 96, 'VCTK_p343': 97, 'VCTK_p345': 98, 'VCTK_p347': 99, 'VCTK_p351': 100, 'VCTK_p360': 101, 'VCTK_p361': 102, 'VCTK_p362': 103, 'VCTK_p363': 104, 'VCTK_p364': 105, 'VCTK_p374': 106, 'VCTK_p376': 107
             }},
        18: {"c_arg": "tts_models/en/sam/tacotron-DDC"},
        19: {"c_arg": "tts_models/en/blizzard2013/capacitron-t2-c50"},
        20: {"c_arg": "tts_models/en/blizzard2013/capacitron-t2-c150_v2"},
        21: {"c_arg": "tts_models/es/mai/tacotron2-DDC"},
        22: {"c_arg": "tts_models/es/css10/vits"},
        23: {"c_arg": "tts_models/fr/mai/tacotron2-DDC"},
        24: {"c_arg": "tts_models/fr/css10/vits"},
        25: {"c_arg": "tts_models/uk/mai/glow-tts"},
        26: {"c_arg": "tts_models/uk/mai/vits"},
        27: {"c_arg": "tts_models/zh-CN/baker/tacotron2-DDC-GST"},
        28: {"c_arg": "tts_models/nl/mai/tacotron2-DDC"},
        29: {"c_arg": "tts_models/nl/css10/vits"},
        30: {"c_arg": "tts_models/de/thorsten/tacotron2-DCA"},
        31: {"c_arg": "tts_models/de/thorsten/vits"},
        32: {"c_arg": "tts_models/de/thorsten/tacotron2-DDC"},
        33: {"c_arg": "tts_models/de/css10/vits-neon"},
        34: {"c_arg": "tts_models/ja/kokoro/tacotron2-DDC"},
        35: {"c_arg": "tts_models/tr/common-voice/glow-tts"},
        36: {"c_arg": "tts_models/it/mai_female/glow-tts"},
        37: {"c_arg": "tts_models/it/mai_female/vits"},
        38: {"c_arg": "tts_models/it/mai_male/glow-tts"},
        39: {"c_arg": "tts_models/it/mai_male/vits"},
        40: {"c_arg": "tts_models/ewe/openbible/vits"},
        41: {"c_arg": "tts_models/hau/openbible/vits"},
        42: {"c_arg": "tts_models/lin/openbible/vits"},
        43: {"c_arg": "tts_models/tw_akuapem/openbible/vits"},
        44: {"c_arg": "tts_models/tw_asante/openbible/vits"},
        45: {"c_arg": "tts_models/yor/openbible/vits"},
        46: {"c_arg": "tts_models/hu/css10/vits"},
        47: {"c_arg": "tts_models/el/cv/vits"},
        48: {"c_arg": "tts_models/fi/css10/vits"},
        49: {"c_arg": "tts_models/hr/cv/vits"},
        50: {"c_arg": "tts_models/lt/cv/vits"},
        51: {"c_arg": "tts_models/lv/cv/vits"},
        52: {"c_arg": "tts_models/mt/cv/vits"},
        53: {"c_arg": "tts_models/pl/mai_female/vits"},
        54: {"c_arg": "tts_models/pt/cv/vits"},
        55: {"c_arg": "tts_models/ro/cv/vits"},
        56: {"c_arg": "tts_models/sk/cv/vits"},
        57: {"c_arg": "tts_models/sl/cv/vits"},
        58: {"c_arg": "tts_models/sv/cv/vits"}
    },
    "vocoder_models": {
        1: {"c_arg": "vocoder_models/universal/libri-tts/wavegrad"},
        2: {"c_arg": "vocoder_models/universal/libri-tts/fullband-melgan"},
        3: {"c_arg": "vocoder_models/en/ek1/wavegrad"},
        4: {"c_arg": "vocoder_models/en/ljspeech/multiband-melgan"},
        5: {"c_arg": "vocoder_models/en/ljspeech/hifigan_v2"},
        6: {"c_arg": "vocoder_models/en/ljspeech/univnet"},
        7: {"c_arg": "vocoder_models/en/blizzard2013/hifigan_v2"},
        8: {"c_arg": "vocoder_models/en/vctk/hifigan_v2"},
        9: {"c_arg": "vocoder_models/en/sam/hifigan_v2"},
        10: {"c_arg": "vocoder_models/nl/mai/parallel-wavegan"},
        11: {"c_arg": "vocoder_models/de/thorsten/wavegrad"},
        12: {"c_arg": "vocoder_models/de/thorsten/fullband-melgan"},
        13: {"c_arg": "vocoder_models/de/thorsten/hifigan_v1"},
        14: {"c_arg": "vocoder_models/ja/kokoro/hifigan_v1"},
        15: {"c_arg": "vocoder_models/uk/mai/multiband-melgan"},
        16: {"c_arg": "vocoder_models/tr/common-voice/hifigan"}
    }
}


def get_from_c_arg(model_c_arg, attr="language"):
    out = None
    c_args = model_c_arg.split("/")
    if attr == "model_type":
        out = c_args[0]
    elif attr == "language":
        out = c_args[1]
    elif attr == "dataset":
        out = c_args[2]
    elif attr == "model_name":
        out = c_args[3]
    return out

