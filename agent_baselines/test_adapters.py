import json
import tempfile
import unittest
from pathlib import Path

from agent_baselines.surgpub import load_surgpub_requests, request_to_medgrpo_record
from agent_baselines.video_io import sample_ranges


class AdapterTest(unittest.TestCase):
    def test_surgpub_medgrpo_record_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.json"
            data_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "q-1",
                            "conversations": [
                                {"from": "human", "value": "<video>\nWhat is the next step?"},
                                {"from": "gpt", "value": "Closure."},
                            ],
                            "video": ["frames/0001.jpg", "frames/0002.jpg"],
                            "metadata": {"fps": 2},
                            "qa_type": "procedure",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            request = load_surgpub_requests(data_path, data_root=directory)[0]
            self.assertEqual(request.question, "What is the next step?")
            self.assertEqual(request.metadata["answer"], "Closure.")
            self.assertEqual(request.frame_paths[0], str(Path(directory) / "frames/0001.jpg"))
            medgrpo = request_to_medgrpo_record(request)
            self.assertEqual(medgrpo["video"], list(request.frame_paths))
            self.assertTrue(medgrpo["conversations"][0]["value"].startswith("<video>\n"))

    def test_long_video_ranges_are_bounded_and_deterministic(self):
        ranges = sample_ranges(
            duration_sec=30,
            window_sec=8,
            stride_sec=4,
            max_windows=3,
        )
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0], (0.0, 8.0))
        self.assertEqual(ranges[-1], (24.0, 30))


if __name__ == "__main__":
    unittest.main()
