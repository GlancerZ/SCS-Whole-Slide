import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scs_parallel_resources as resources


class ResourceTests(unittest.TestCase):
    def test_pid_identity_is_live(self):
        self.assertIsNotNone(resources.identity(os.getpid()))
        self.assertIsNone(resources.identity(999999999))

    def test_lease_grants_and_cleans_up_on_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = dict(memory_gib=100, gpus=[dict(uuid="GPU-test", total=80000)],
                          max_trainers_per_gpu=4, end_time=9999999999)
            with patch.object(resources, "configure", return_value=config), \
                 patch.object(resources, "runtime", return_value=root), \
                 patch.object(resources, "legacy_usage", return_value=0), \
                 patch.object(resources, "physical_pressure", return_value=0), \
                 patch.object(resources, "gpu_rows", return_value=[dict(uuid="GPU-test", used=0)]), \
                 patch.object(resources, "estimate", return_value=(32, 16384)), \
                 patch.dict(os.environ):
                with self.assertRaisesRegex(RuntimeError, "test-body"):
                    with resources.lease("train", root / "tile") as record:
                        self.assertEqual(record["gpu"], "GPU-test")
                        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-test")
                        saved = json.loads((root / "leases.json").read_text())
                        self.assertTrue(saved[str(os.getpid())]["active"])
                        raise RuntimeError("test-body")
                self.assertEqual(json.loads((root / "leases.json").read_text()), {})

    def test_oversized_stage_fails_without_running(self):
        with patch.object(resources, "configure", return_value=dict(memory_gib=100)), \
             patch.object(resources, "estimate", return_value=(120, 0)):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                with resources.lease("preprocess", Path("tile")):
                    self.fail("Oversized stage was granted")

    def test_existing_reservations_block_ram_and_gpu_oversubscription(self):
        for memory, gpu in [(80, 16384), (20, 70000)]:
            with self.subTest(memory=memory, gpu=gpu), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                other = dict(identity="alive", active=True, memory_gib=memory,
                             gpu="GPU-test", gpu_limit_mib=gpu)
                (root / "leases.json").write_text(json.dumps({"other": other}))
                config = dict(memory_gib=100, gpus=[dict(uuid="GPU-test", total=80000)],
                              max_trainers_per_gpu=4, end_time=0)
                with patch.object(resources, "configure", return_value=config), \
                     patch.object(resources, "runtime", return_value=root), \
                     patch.object(resources, "identity", return_value="alive"), \
                     patch.object(resources, "legacy_usage", return_value=0), \
                     patch.object(resources, "physical_pressure", return_value=0), \
                     patch.object(resources, "gpu_rows", return_value=[dict(uuid="GPU-test", used=0)]), \
                     patch.object(resources, "estimate", return_value=(32, 16384)):
                    with self.assertRaises(TimeoutError):
                        with resources.lease("train", root / "tile"):
                            self.fail("Oversubscribed stage was granted")
                    self.assertEqual(json.loads((root / "leases.json").read_text()), {"other": other})


if __name__ == "__main__":
    unittest.main()
