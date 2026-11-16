"""amerop: American option pricing with binomial trees, Longstaff-Schwartz LSMC,
machine-learned continuation values and reinforcement learning (LSPI)."""

from .models import OptionSpec, bs_price, simulate_gbm
from .binomial import crr, bbsr, bermudan_tree
from .longstaff_schwartz import LSMCConfig, lsmc, evaluate_policy, exercise_boundary
from .rl import lspi, fqi

__all__ = ["OptionSpec", "bs_price", "simulate_gbm", "crr", "bbsr", "bermudan_tree",
           "LSMCConfig", "lsmc", "evaluate_policy", "exercise_boundary", "lspi", "fqi"]
