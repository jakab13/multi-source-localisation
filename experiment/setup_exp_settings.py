from labplatform.config import *

# load default settings
_settings, __changed_settings = load_settings()

set_config("EXPERIMENTER", ["Jakab, Nathalie, Carsten, Max"])
set_config("PARADIGM", ["Numerosity_Judgement", "Spatial_Unmasking"])

save_env_config("MSL")