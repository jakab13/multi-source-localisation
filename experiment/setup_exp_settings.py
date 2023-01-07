from labplatform.config import *

# TODO: get_config does not load custom config settings

if __name__ == "__main__":
    # load default settings
    _settings, __changed_settings = load_settings()

    # change config params
    set_config("EXPERIMENTER", ["Jakab, Nathalie, Carsten, Max"])
    set_config("PARADIGM", ["Numerosity_Judgement", "Spatial_Unmasking"])

    # save new config file
    save_env_config("MSL")

    # load custom config
    DIR = get_config("SETTINGS_ROOT")
    fp = os.path.join(DIR, os.listdir(DIR)[0])
    settings, changed_settings = load_settings(fp)
    setting = get_config(setting="PARADIGM")
