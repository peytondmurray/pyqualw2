import os
import subprocess
import tempfile
import time
from pathlib import Path


class ModelRunner:
    """A class for exceuting the Qualw2 model."""

    def __init__(self, configuration: dict, output_dir: Path):
        self.config = configuration
        self.output_dir = output_dir
        self.model_dir = configuration["model dir"]
        self.wait_time = configuration["wait time"]
        self.run_name = configuration["run name"]

    def run(self):
        """Run the model for each configuration in the list."""
        wd = self.make_temp_wd()
        self.copy_config_files(self.model_dir, wd)
        self.run_model(wd, self.wait_time)
        run_output_dir = self.output_dir / str(self.run_name)
        self.save_outputs(wd, run_output_dir)

        return

    def make_temp_wd(self) -> Path:
        """Make a temporary working directory."""
        return Path(tempfile.mkdtemp())

    def copy_config_files(self, model_dir: Path, wd: Path):
        """Copy the configuration files to the working directory."""
        wd = Path(wd)
        src = model_dir

        if not src.exists():
            raise FileNotFoundError

        for file in src.iterdir():
            destination = wd / file.name
            file.copy(destination, preserve_metadata=True)  # type: ignore

    def save_outputs(self, wd: Path, output_dir: Path):
        """Save the output files to the output directory."""
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        for file in wd.iterdir():
            file.copy(output_dir / file.name, preserve_metadata=True)  # type: ignore

        return

    def run_model(self, wd: Path, wait_time: int):
        """Run the model for the given configuration."""
        model_path = wd / "w2_v45_64.exe"
        try:
            if os.name == "nt":  # Windows
                creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                process = subprocess.Popen(
                    ["w2_v45_64.exe"], creationflags=creationflags
                )
            else:  # Linux/Unix
                process = subprocess.Popen(
                    ["wine", str(model_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(wd),
                )

            # List of output files to monitor
            output_files = ["two_31.csv", "qwo_31.csv", "tsr_1_seg31.csv"]

            # Wait 20 seconds before starting to check file activity
            time.sleep(10)
            # Grab time model strated
            start_time = time.time()

            # Track the last time any file was modified
            last_activity_time = time.time()

            while process.poll() is None and (time.time() - start_time) < wait_time:
                # Check if any file has been modified recently
                any_file_active = False
                for file in output_files:
                    filepath = wd / file
                    if os.path.exists(filepath) and self.check_file_activity(filepath):
                        any_file_active = True
                        last_activity_time = time.time()
                        break
                if any_file_active:
                    time.sleep(5)
                else:
                    # If no files have been modified for 10 seconds,
                    # consider it complete
                    if time.time() - last_activity_time > 10:
                        # Close the model window
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/F", "/IM", "w2_v45_64.exe"])
                        else:
                            process.kill()
                        return
                    else:
                        time.sleep(5)

            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/IM", "w2_v45_64.exe"])
                else:
                    process.kill()
                raise Exception("Simulation timed out")

        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/IM", "w2_v45_64.exe"])
            else:
                process.kill()
            raise
        except Exception:
            raise

    def check_file_activity(self, file_path: Path, timeout: int = 10) -> bool:
        """Check if the file has been modified within the timeout period."""
        try:
            if not os.path.exists(file_path):
                return False

            # Get the current time and file's last modification time
            current_time = time.time()
            file_mtime = os.path.getmtime(file_path)

            # Check if the file has been modified within the timeout period
            time_since_mod = current_time - file_mtime
            return time_since_mod < timeout
        except Exception:
            return False
