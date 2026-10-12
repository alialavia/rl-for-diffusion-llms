import os

from hypothesis import HealthCheck, settings

# Profiles (pbt-hypothesis settings.md): "dev" is the default, "ci" is quick,
# "thorough" is for occasional deep runs: HYPOTHESIS_PROFILE=thorough pytest
_common = dict(deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", max_examples=40, **_common)
settings.register_profile("ci", max_examples=20, derandomize=True, **_common)
settings.register_profile("thorough", max_examples=500, **_common)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
