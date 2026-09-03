import unittest

try:
    from .controller import (
        ClipCandidate,
        InspectionResult,
        ModelAnswer,
        SurgicalController,
        VideoRequest,
    )
except ImportError:  # Support running this file from inside agent_baselines.
    from controller import (
        ClipCandidate,
        InspectionResult,
        ModelAnswer,
        SurgicalController,
        VideoRequest,
    )


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


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.request = VideoRequest("q1", "v1", "What happened after cutting?")

    def test_direct_does_not_retrieve(self):
        model = FakeAnswerModel()
        retriever = FakeRetriever()
        output = SurgicalController(model, retriever=retriever, mode="direct").run(self.request)
        self.assertEqual(output.answer.text, "draft answer")
        self.assertEqual(retriever.calls, 0)

    def test_retrieval_uses_top1(self):
        model = FakeAnswerModel()
        retriever = FakeRetriever()
        output = SurgicalController(model, retriever=retriever, mode="retrieve").run(self.request)
        self.assertEqual(output.trace["retrieved"][0]["clip_id"], "top")
        self.assertEqual(model.calls[0][0].clip_id, "top")

    def test_adaptive_calls_inspector_once_and_rethinks(self):
        model = FakeAnswerModel()
        retriever = FakeRetriever()
        inspector = FakeInspector()
        output = SurgicalController(
            model,
            retriever=retriever,
            inspector=inspector,
            mode="adaptive",
        ).run(self.request)
        self.assertEqual(output.answer.text, "final answer")
        self.assertEqual(inspector.calls, 1)
        self.assertEqual(len(output.trace["tool_calls"]), 1)
        self.assertEqual(output.trace["tool_calls"][0]["output"], "inspection evidence")
        self.assertEqual(output.trace["answer_stage"], "rethink_after_inspection")


if __name__ == "__main__":
    unittest.main()
