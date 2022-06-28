"""
Federated learning-based adversaries.  
The `convert` function is used to convert an existing endpoint object (Scout) into the specified adversary.  
The `GradientTransform` classes perform the adversarial activity at the network level,
and are to be added to the network controller.
"""


from . import alternating_minimization
from . import backdoor
from . import scaler
from . import onoff
from . import freerider
from . import mouther
from . import labelflipper