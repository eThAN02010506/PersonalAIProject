import unittest

from qwopus_agent.agent import AgentLoop, Executor, Planner


class AgentLoopTests(unittest.TestCase):
    def test_agent_loop_runs_minimal_plan(self) -> None:
        agent = AgentLoop(planner=Planner(), executor=Executor())

        result = agent.run("Create a project skeleton")

        self.assertEqual(result.objective, "Create a project skeleton")
        self.assertTrue(result.execution.success)
        self.assertIn("Create a project skeleton", result.execution.output)
