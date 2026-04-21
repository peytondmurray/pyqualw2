import os
import subprocess
import tempfile
import time
from pathlib import Path

from pyqualw2.config.config import Config


class ModelRunner:
    """A class for exceuting the Qualw2 model.

    Parameters
    ----------
    configuration : Config
        The configuration to run the model with.
    output_dir : Path
        The directory to save the output files to.
    source_dir : Path
        The directory to read the source files from.
    run_name : str
        The name of the run.
    wait_time : int, optional
        The time to wait for the model to finish running, by default 30 seconds.
    """

    def __init__(
        self,
        configuration: Config,
        output_dir: Path,
        source_dir: Path,
        run_name: str,
        wait_time: int = 30,
    ):
        self.config = configuration
        self.output_dir = output_dir
        self.source_dir = source_dir
        self.run_name = run_name
        self.wait_time = wait_time if wait_time is not None else 30

    def run(self):
        """Run the model for each configuration in the list."""
        wd = self.make_temp_wd()
        self.config.to_directory(wd, overwrite=True)

        # Make the outputs directory, otherwise cequalw2 will fail
        (wd / "outputs").mkdir(exist_ok=True)
        self.run_model(wd)
        run_output_dir = self.output_dir / self.run_name
        self.save_outputs(wd, run_output_dir)

    def make_temp_wd(self) -> Path:
        """Make a temporary working directory to run cequalw2 in.

        Returns
        -------
        Path
            Path to the directory
        """
        return Path(tempfile.mkdtemp())

    def save_outputs(self, wd: Path, output_dir: Path):
        """Copy the cequalw2-generated outputs to the output directory.

        Parameters
        ----------
        wd : Path
            Working directory where cequalw2 is run
        output_dir : Path
            Directory where the user has asked for results to be placed
        """
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        for file in wd.iterdir():
            file.copy(output_dir / file.name, preserve_metadata=True)  # type: ignore

        return

    def run_model(self, wd: Path):
        """Run the cequalw2 binary in a subprocess.

        Parameters
        ----------
        wd : Path
            Path to the working directory. Must contain the cequalw2 binary

        Raises
        ------
        TimeoutExpired
            Never actually raised. See
            https://github.com/steelhead-dev/pyqualw2/issues/59 for the tracking issue.
        Exception
            Possibly not raised? Not sure what can fail here. See the issue above.
        """
        model_path = wd / "w2_v45_64.exe"
        try:
            if os.name == "nt":  # Windows
                creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                process = subprocess.Popen(
                    [str(model_path)],
                    creationflags=creationflags,
                    cwd=str(wd),
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

            while (
                process.poll() is None and (time.time() - start_time) < self.wait_time
            ):
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
        """Check if the file has been modified within the timeout period.

        Parameters
        ----------
        file_path : Path
            Path to the file to check for modifications
        timeout : int
            Number of seconds to consider when checking for activity

        Returns
        -------
        bool
            True if the file has been modified in the last `timeout` seconds, False
            otherwise (or if the file doesn't exist)
        """
        if not os.path.exists(file_path):
            return False

        # Get the current time and file's last modification time
        current_time = time.time()
        file_mtime = os.path.getmtime(file_path)

        # Check if the file has been modified within the timeout period
        time_since_mod = current_time - file_mtime
        return time_since_mod < timeout
