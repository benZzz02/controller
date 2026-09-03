import unittest

from agent_baselines.controller import (
    ClipCandidate,
    InspectionResult,
    ModelAnswer,
    VideoRequest,
)
from agent_baselines.langgraph_controller import LangGraphSurgicalController

try:
    import langgraph  # noqa: F401
except ImportError:
    HAS_LANGGRAPH = False
else:
    HAS_LANGGRAPH = True


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, request, top_k):
        self.calls += 1
        return [
            ClipCandidate("top", request.video_id, 0.0, 10.0, 0.50),
            ClipCandidate("second", request.video_id, 10.0, 20.0, 0.49),
        ][:top_k]


class FakeAnswerModel:
    def __init__(self):
        self.calls = []

    def answer(self, request, candidate=None, evidence=None, draft=None):
        self.calls.append((candidate, evidence, draft))
        if evidence is not None:
            return ModelAnswer("final answer", confidence=0.9)
        return ModelAnswer("draft answer", confidence=0.8)


class FakeInspector:
    def __init__(self):
        self.calls = 0

    def inspect(self, request, candidate):
        self.calls += 1
        return InspectionResult("inspection evidence")


@unittest.skipUnless(HAS_LANGGRAPH, "langgraph is optional in the lightweight test environment")
class LangGraphControllerTest(unittest.TestCase):
    def setUp(self):
        self.request = VideoRequest("q1", "v1", "What happened after cutting?")

    def test_adaptive_calls_inspector_once_and_records_graph_trace(self):
        model = FakeAnswerModel()
        retriever = FakeRetriever()
        inspector = FakeInspector()
        controller = LangGraphSurgicalController(
            model,
            retriever=retriever,
            inspector=inspector,
            mode="adaptive",
        )

        output = controller.run(self.request)

        self.assertEqual(output.answer.text, "final answer")
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(inspector.calls, 1)
        self.assertEqual(len(output.trace["tool_calls"]), 1)
        self.assertEqual(output.trace["framework"], "langgraph")
        self.assertEqual(output.trace["answer_stage"], "rethink_after_inspection")
        self.assertIn("retrieve", output.trace["stages"])
        self.assertIn("gate", output.trace["stages"])

    def test_direct_path_does_not_call_retriever(self):
        model = FakeAnswerModel()
        retriever = FakeRetriever()
        output = LangGraphSurgicalController(
            model,
            retriever=retriever,
            mode="direct",
        ).run(self.request)

        self.assertEqual(output.answer.text, "draft answer")
        self.assertEqual(retriever.calls, 0)
        self.assertEqual(output.trace["answer_stage"], "direct")

    def test_retrieve_path_uses_top1_without_inspection(self):
        model = FakeAnswerModel()
        retriever = FakeRetriever()
        inspector = FakeInspector()
        output = LangGraphSurgicalController(
            model,
            retriever=retriever,
            inspector=inspector,
            mode="retrieve",
        ).run(self.request)

        self.assertEqual(output.answer.text, "draft answer")
        self.assertEqual(model.calls[0][0].clip_id, "top")
        self.assertEqual(inspector.calls, 0)
        self.assertEqual(output.trace["answer_stage"], "retrieved_top1")

    def test_inspect_path_returns_tool_text_without_controller_answer(self):
        model = FakeAnswerModel()
        retriever = FakeRetriever()
        inspector = FakeInspector()
        output = LangGraphSurgicalController(
            model,
            retriever=retriever,
            inspector=inspector,
            mode="inspect",
        ).run(self.request)

        self.assertEqual(output.answer.text, "inspection evidence")
        self.assertEqual(model.calls, [])
        self.assertEqual(inspector.calls, 1)
        self.assertEqual(output.trace["answer_stage"], "inspector")

    def test_stream_exposes_node_updates(self):
        controller = LangGraphSurgicalController(
            FakeAnswerModel(),
            retriever=FakeRetriever(),
            inspector=FakeInspector(),
            mode="adaptive",
        )

        updates = list(controller.stream(self.request))
        nodes = {node for update in updates for node in update}

        self.assertTrue({"retrieve", "draft", "gate", "inspect", "final_answer"}.issubset(nodes))


if __name__ == "__main__":
    unittest.main()
