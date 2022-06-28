# Regiment
Endpoint side functionalities, inclusive of [adversaries](regiment/adversaries).
The [Scout](regiment/scout) object is the basic/standard endpoint
collaborator for federated learning, while the adversary modules each act either as modifiers for [Scout](regiment/scout)
instances (with the function `convert`) or as update transforms at the network level (with the `GradientTransform` objects).