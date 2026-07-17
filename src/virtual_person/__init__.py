"""Python-native embodied virtual-person simulation."""

from .agent import VirtualPerson
from .env import VirtualPersonEnv
from .simulation import VirtualPersonSimulation
from .real_world import (
    ActuatorSuite,
    EmergencyStop,
    MockActuators,
    MockSensors,
    RealWorldConfig,
    RealWorldRuntime,
    SafetyPolicy,
    SensorSnapshot,
    SensorSuite,
)
from .types import Action, ActionKind, ActionResult, Goal, GoalKind

from .drives import CognitiveDrives, SatisfactionEvaluator
from .interoception import (
    DEFAULT_DRIVE_SOURCES,
    DEFAULT_LEVELS,
    DedicatedDriveNeuronBank,
    DriveNeuronSpec,
    DriveSourceSpec,
)
from .spike_tokenizer import ByteTokenizer
from .spiking import (
    ClusterState,
    LinkSpec,
    ModelState,
    NodeLinkSpikeCluster,
    NodeLinkSpikeModel,
    NodeSpec,
    SpikingModelConfig,
    SpikingOutput,
)
from .spiking_mind import MindDecision, SpikingMind
from .spiking_runtime import AutonomousSpikingAgent, AutonomousStep

__all__ = [
    "Action",
    "ActionKind",
    "ActionResult",
    "Goal",
    "GoalKind",
    "VirtualPerson",
    "VirtualPersonEnv",
    "VirtualPersonSimulation",
    "SpikingOutput",
    "SpikingModelConfig",
    "SpikingMind",
    "SatisfactionEvaluator",
    "NodeSpec",
    "NodeLinkSpikeModel",
    "NodeLinkSpikeCluster",
    "ModelState",
    "MindDecision",
    "LinkSpec",
    "CognitiveDrives",
    "DriveSourceSpec",
    "DriveNeuronSpec",
    "DedicatedDriveNeuronBank",
    "DEFAULT_LEVELS",
    "DEFAULT_DRIVE_SOURCES",
    "ClusterState",
    "ByteTokenizer",
    "AutonomousStep",
    "AutonomousSpikingAgent",
    "ActuatorSuite",
    "EmergencyStop",
    "MockActuators",
    "MockSensors",
    "RealWorldConfig",
    "RealWorldRuntime",
    "SafetyPolicy",
    "SensorSnapshot",
    "SensorSuite",
]

__version__ = "0.5.0"
