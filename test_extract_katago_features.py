import unittest
import subprocess
import os
import sys
import urllib.request
import zipfile
import shutil
import requests

class TestFeatureExtractorCLI(unittest.TestCase):
    SCRIPT_PATH = 'extract_katago_features.py'
    # Use a tiny 2-block model for testing to speed things up.
    # This is an official model from the KataGo training repository (media.katagotraining.org),
    # but its small size makes it ideal for fast, automated tests.
    TEST_MODEL_URL = "https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b2c96-s133333504-d2533815.zip"
    TEST_MODEL_OUTPUT_SHAPE = "(96, 19, 19)"
    model_zip_path = None
    model_dir_path = None
    model_ckpt_path = None
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

        # Download and extract the test model once for all tests to ensure speed.
        print(f"Setting up test suite: downloading model from {cls.TEST_MODEL_URL}")
        cls.model_zip_path = os.path.basename(cls.TEST_MODEL_URL)
        model_dir_name = os.path.splitext(cls.model_zip_path)[0]
        cls.model_dir_path = model_dir_name
        cls.model_ckpt_path = os.path.join(model_dir_name, "model.ckpt")

        if not os.path.exists(cls.model_ckpt_path):
            # Use a more modern-looking User-Agent to avoid 403 Forbidden errors.
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            try:
                response = requests.get(cls.TEST_MODEL_URL, headers=headers, stream=True)
                response.raise_for_status()
                with open(cls.model_zip_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
            except requests.exceptions.RequestException as e:
                # In a test setup, we want to fail fast and clearly.
                cls.tearDownClass() # Clean up what we can
                raise RuntimeError(f"Failed to download test model: {e}") from e

            with zipfile.ZipFile(cls.model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
        print("Model setup complete.")

    @classmethod
    def tearDownClass(cls):
        # Clean up the dummy SGF files
        for filename in [cls.TEST_SGF_FILENAME, cls.TEST2_SGF_FILENAME]:
            if os.path.exists(filename):
                os.remove(filename)

        # Clean up model files
        if cls.model_zip_path and os.path.exists(cls.model_zip_path):
            os.remove(cls.model_zip_path)
        if cls.model_dir_path and os.path.exists(cls.model_dir_path):
            shutil.rmtree(cls.model_dir_path)

    def run_script(self, args):
        """Helper method to run the script with given arguments."""
        command = [sys.executable, self.SCRIPT_PATH] + args
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_single_node(self):
        """Test extracting features for a single node."""
        args = ['--sgf-node', self.TEST_SGF_FILENAME, "0,0", '--model-path', self.model_ckpt_path]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn(f"--- Features for {self.TEST_SGF_FILENAME} at path '0,0' ---", result.stdout)
        self.assertIn(f"Trunkfinal output shape: {self.TEST_MODEL_OUTPUT_SHAPE}", result.stdout)

    def test_batch_processing(self):
        """Test batch processing of multiple nodes."""
        args = [
            '--model-path', self.model_ckpt_path,
            '--sgf-node', self.TEST_SGF_FILENAME, "",
            '--sgf-node', self.TEST2_SGF_FILENAME, "0,0,1"
        ]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn(f"--- Features for {self.TEST_SGF_FILENAME} at path 'root' ---", result.stdout)
        self.assertIn(f"--- Features for {self.TEST2_SGF_FILENAME} at path '0,0,1' ---", result.stdout)
        self.assertEqual(result.stdout.count(f"Trunkfinal output shape:"), 2)

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
        args = ['--model-path', self.model_ckpt_path]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn("--- Using initial demo game state ---", result.stdout)
        self.assertIn(f"Trunkfinal output shape: {self.TEST_MODEL_OUTPUT_SHAPE}", result.stdout)

if __name__ == '__main__':
    unittest.main()
