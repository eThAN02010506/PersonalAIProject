"""Public API for supervised multi-agent orchestration."""

from qwopus_agent.agents.multi_agent.arbitration import (
    ConsensusArbiter,
    LLMResultArbiter,
)
from qwopus_agent.agents.multi_agent.delegation import (
    DeterministicTaskDelegator,
    LLMTaskDelegator,
)
from qwopus_agent.agents.multi_agent.models import (
    AgentContribution,
    AgentProfile,
    ArbitrationDecision,
    DebateStatement,
    DelegatedTask,
    DelegationPlan,
    MultiAgentRun,
    NamedAgentRun,
    ResultArbiter,
    RunnableAgent,
    TaskDelegator,
)
from qwopus_agent.agents.multi_agent.state import SharedAgentState
from qwopus_agent.agents.multi_agent.supervisor import (
    MultiAgentCoordinator,
    MultiAgentSupervisor,
)

__all__ = [
    "AgentContribution",
    "AgentProfile",
    "ArbitrationDecision",
    "ConsensusArbiter",
    "DebateStatement",
    "DelegatedTask",
    "DelegationPlan",
    "DeterministicTaskDelegator",
    "LLMResultArbiter",
    "LLMTaskDelegator",
    "MultiAgentCoordinator",
    "MultiAgentRun",
    "MultiAgentSupervisor",
    "NamedAgentRun",
    "ResultArbiter",
    "RunnableAgent",
    "SharedAgentState",
    "TaskDelegator",
]
