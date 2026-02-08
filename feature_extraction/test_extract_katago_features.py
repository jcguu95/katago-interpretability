import unittest
import subprocess
import os
import sys
import urllib.request
import zipfile
import shutil
import requests

class TestFeatureExtractorCLI(unittest.TestCase):
    SCRIPT_PATH = 'feature_extraction/extract_katago_features.py'
    # Use a tiny 2-block model for testing to speed things up.
    # This is an official model from the KataGo training repository (media.katagotraining.org),
    # but its small size makes it ideal for fast, automated tests. The previous URL became
    # inaccessible, so this has been updated to a different small model (28-block).
    TEST_MODEL_URL = "https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b28c512nbt-s12404017920-d5711392113.zip"
    TEST_MODEL_OUTPUT_SHAPE = "(512, 19, 19)"
    model_zip_path = None
    model_dir_path = None
    model_file_path = None
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
        print(f"Setting up test suite...")
        cls.model_zip_path = os.path.basename(cls.TEST_MODEL_URL)

        if not os.path.exists(cls.model_zip_path):
            if os.environ.get('ALLOW_MODEL_DOWNLOAD') != '1':
                raise FileNotFoundError(
                    f"Test model '{cls.model_zip_path}' not found. "
                    "Run 'make test-full' to download it."
                )
            
            print(f"Test model not found. Downloading from {cls.TEST_MODEL_URL}")
            # Use a more modern-looking User-Agent and other headers to avoid 403 Forbidden errors.
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://katagotraining.org/'
            }
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
        
        # Determine model path from zip contents and extract if necessary.
        with zipfile.ZipFile(cls.model_zip_path, 'r') as zip_ref:
            model_file_zip_path = None
            # Find the full path to model file within the zip archive
            for name in zip_ref.namelist():
                if name.endswith('.ckpt'):
                    model_file_zip_path = name
                    break

            if not model_file_zip_path:
                contents = "\n".join(zip_ref.namelist())
                raise RuntimeError(f"Could not find model file (.ckpt) in the zip: {cls.model_zip_path}\nContents:\n{contents}")
            
            # The directory path is the parent of model file
            model_dir_name = os.path.dirname(model_file_zip_path)
            cls.model_dir_path = os.path.abspath(model_dir_name)
            cls.model_file_path = os.path.abspath(model_file_zip_path)

            if not os.path.exists(cls.model_dir_path) or not os.path.exists(cls.model_file_path):
                print(f"Extracting model...")
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
        args = ['--sgf-node', self.TEST_SGF_FILENAME, "0,0", '--model-path', self.model_file_path]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn(f"--- Features for {self.TEST_SGF_FILENAME} at path '0,0' ---", result.stdout)
        self.assertIn(f"Trunkfinal output shape: {self.TEST_MODEL_OUTPUT_SHAPE}", result.stdout)

    def test_batch_processing(self):
        """Test batch processing of multiple nodes."""
        args = [
            '--model-path', self.model_file_path,
            '--sgf-node', self.TEST_SGF_FILENAME, "",
            '--sgf-node', self.TEST2_SGF_FILENAME, "0,0,0"
        ]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn(f"--- Features for {self.TEST_SGF_FILENAME} at path 'root' ---", result.stdout)
        self.assertIn(f"--- Features for {self.TEST2_SGF_FILENAME} at path '0,0,0' ---", result.stdout)
        self.assertEqual(result.stdout.count(f"Trunkfinal output shape:"), 2)

    def test_invalid_variation_path(self):
        """Test with an invalid variation path."""
        args = ['--model-path', self.model_file_path, '--sgf-node', self.TEST2_SGF_FILENAME, "0,1"]
        result = self.run_script(args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid variation path: branch index 1 is out of range", result.stderr)

    def test_missing_sgf_file(self):
        """Test with a non-existent SGF file."""
        args = ['--model-path', self.model_file_path, '--sgf-node', 'non_existent_file.sgf', ""]
        result = self.run_script(args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SGF file not found: non_existent_file.sgf", result.stderr)

    def test_no_args(self):
        """Test running the script with no SGF arguments, which should run the demo."""
        args = ['--model-path', self.model_file_path]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertIn("--- Using initial demo game state ---", result.stdout)
        self.assertIn(f"Trunkfinal output shape: {self.TEST_MODEL_OUTPUT_SHAPE}", result.stdout)

if __name__ == '__main__':
    unittest.main()
