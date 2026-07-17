"""Python-native embodied virtual-person simulation."""

from .agent import VirtualPerson
from .economy import AssignedTask, TaskEconomy, TaskReview, TaskStatus
from .env import VirtualPersonEnv
from .renderer import Renderer, RendererConfig
from .simulation import VirtualPersonSimulation
from .vm_runtime import QemuActuators, QemuSensors, QemuVirtualMachine, QemuVmConfig, QmpClient
from .vp_package import VpContents, build_training_manifest, inspect as inspect_vp, pack as pack_vp, unpack as unpack_vp
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
    "Renderer",
    "RendererConfig",
    "TaskEconomy",
    "AssignedTask",
    "TaskReview",
    "TaskStatus",
    "pack_vp",
    "unpack_vp",
    "inspect_vp",
    "VpContents",
    "build_training_manifest",
    "QemuVirtualMachine",
    "QemuVmConfig",
    "QemuSensors",
    "QemuActuators",
    "QmpClient",
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
