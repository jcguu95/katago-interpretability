import unittest
import subprocess
import os
import sys

class TestFeatureExtractorCLI(unittest.TestCase):
    SCRIPT_PATH = 'extract_katago_features.py'
    TEST_SGF_CONTENT = "(;GM[1]SZ[19];B[aa];W[bb])"
    TEST2_SGF_CONTENT = "(;GM[1]SZ[19];B[dd];W[pp];B[dp])"
    TEST_SGF_FILENAME = "test.sgf"
    TEST2_SGF_FILENAME = "test2.sgf"

    @classmethod
    def setUpClass(cls):
        # Create dummy test SGF files for tests
        with open(cls.TEST_SGF_FILENAME, "w") as f:
            f.write(cls.TEST_SGF_CONTENT)
        with open(cls.TEST2_SGF_FILENAME, "w") as f:
            f.write(cls.TEST2_SGF_CONTENT)

    @classmethod
    def tearDownClass(cls):
        # Clean up the dummy SGF files
        for filename in [cls.TEST_SGF_FILENAME, cls.TEST2_SGF_FILENAME]:
            if os.path.exists(filename):
                os.remove(filename)

    def run_script(self, args):
        """Helper method to run the script with given arguments."""
        command = [sys.executable, self.SCRIPT_PATH] + args
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_single_node(self):
        """Test extracting features for a single node."""
        args = ['--sgf-node', self.TEST_SGF_FILENAME, "0,0"]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn(f"--- Features for {self.TEST_SGF_FILENAME} at path '0,0' ---", result.stdout)
        self.assertIn("Trunkfinal output shape: (512, 19, 19)", result.stdout)

    def test_batch_processing(self):
        """Test batch processing of multiple nodes."""
        args = [
            '--sgf-node', self.TEST_SGF_FILENAME, "",
            '--sgf-node', self.TEST2_SGF_FILENAME, "0,0,1"
        ]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn(f"--- Features for {self.TEST_SGF_FILENAME} at path 'root' ---", result.stdout)
        self.assertIn(f"--- Features for {self.TEST2_SGF_FILENAME} at path '0,0,1' ---", result.stdout)
        self.assertEqual(result.stdout.count("Trunkfinal output shape:"), 2)

    def test_invalid_variation_path(self):
        """Test with an invalid variation path."""
        args = ['--sgf-node', self.TEST2_SGF_FILENAME, "0,1"]
        result = self.run_script(args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid variation path: branch index 1 is out of range", result.stderr)

    def test_missing_sgf_file(self):
        """Test with a non-existent SGF file."""
        args = ['--sgf-node', 'non_existent_file.sgf', ""]
        result = self.run_script(args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SGF file not found: non_existent_file.sgf", result.stderr)

    def test_no_args(self):
        """Test running the script with no SGF arguments, which should run the demo."""
        result = self.run_script([])
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn("--- Using initial demo game state ---", result.stdout)
        self.assertIn("Trunkfinal output shape: (512, 19, 19)", result.stdout)

if __name__ == '__main__':
    unittest.main()
