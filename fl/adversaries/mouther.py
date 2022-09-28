"""
The bad and good mouthing attacks for federated learning.
"""


import fl.utils.functions


class GradientTransform:
    """
    Gradient transformation that copies the gradient of the victim client to the adversaries.
    """
    def __init__(self, num_adversaries, victim, attack_type):
        """
        Construct the gradient transformation.

        Arguments:
        - num_adversaries: the number of adversaries
        - victim: the index of the victim client
        - attack_type: the attack type to use, options are "bad" and "good" for the bad and good mouthing attacks respectively
        """
        self.num_adv = num_adversaries
        self.victim = victim
        self.attack_type = attack_type
        
    def __call__(self, all_grads):
        """Copy victim gradient to all adversaries, negate the adversary gradients if bad mouthing."""
        grad = all_grads[self.victim]
        if "bad" in self.attack_type:
            grad = fl.utils.functions.tree_mul(grad, -1)
        all_grads[-self.num_adv:] = [
            fl.utils.functions.tree_add_normal(grad, loc=0.0, scale=10e-4) for _ in range(self.num_adv)
        ]
        return all_grads
