from labplatform.config import *

# TODO: get_config does not load custom config settings
# load default settings
_settings, __changed_settings = load_settings()

set_config("EXPERIMENTER", ["Jakab, Nathalie, Carsten, Max"])
set_config("PARADIGM", ["Numerosity_Judgement", "Spatial_Unmasking"])

save_env_config("MSL")

if __name__ == "__main__":
    DIR = _settings["SETTINGS_ROOT"]
    settings = load_settings(DIR)
    setting = get_config()
