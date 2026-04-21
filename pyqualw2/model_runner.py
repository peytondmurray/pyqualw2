import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from pyqualw2.config.config import Config

log = logging.getLogger(__name__)


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
        wait_time: int = 10,
    ):
        self.config = configuration
        self.output_dir = output_dir
        self.source_dir = source_dir
        self.run_name = run_name
        self.wait_time = wait_time

    def run(self):
        """Run the model for each configuration in the list.

        Raises
        ------
        ValueError
            Raised if the cequalw2 subprocess returned nonzero
        """
        wd = Path(tempfile.mkdtemp())
        # Make the outputs directory, otherwise cequalw2 will fail
        (wd / "outputs").mkdir(exist_ok=True)

        self.config.to_directory(wd, overwrite=True)
        retcode = self.run_model(wd)
        if retcode is None or retcode == 0:
            # The subprocess return code is None if it was terminated early (i.e. the
            # window was still open, as it always is when the simulation completes)
            self.save_outputs(wd, self.output_dir / self.run_name)
        else:
            raise ValueError(f"cequalw2 subprocess returned error code {retcode}")

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
            file.copy(output_dir / file.name, preserve_metadata=True)

    def run_model(self, wd: Path) -> int:
        """Run the cequalw2 binary in a subprocess.

        Parameters
        ----------
        wd : Path
            Path to the working directory. Must contain the cequalw2 binary

        Returns
        -------
        int
            The return code of the cequalw2 subprocess

        Raises
        ------
        ValueError
            Raised if the cequalw2 subprocess doesn't have a valid stdout/stderr
        """
        model_path = wd / "w2_v45_64.exe"
        if os.name == "nt":
            cmd = [str(model_path)]
        else:
            cmd = ["wine", str(model_path)]

        # List of output files to monitor
        output_files = ["two_31.csv", "qwo_31.csv", "tsr_1_seg31.csv"]

        last_access = time.time()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(wd)
        )
        if process.stdout is None or process.stderr is None:
            raise ValueError("Cannot poll subprocess if stdout or stderr is None")

        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        log.info(f"Launched process {process.pid} in working directory {str(wd)}...")
        # Check for writes to output files; if the last write is longer than
        # self.wait_time seconds ago, consider the simulation done. Kill the
        # subprocess and return.
        while process.poll() is None:
            # If the output files are present, update the last access time if it's
            # more recent that the current last access time
            for file in output_files:
                path = wd / "outputs" / file
                if path.exists():
                    last_access = max(last_access, path.stat().st_mtime)

            log.info(
                f"Polling process {process.pid}. Last observed change "
                f"{time.time() - last_access:.2f}/{self.wait_time} ago..."
            )
            out = process.stdout.readline()
            err = process.stderr.readline()
            if out:
                log.info(f"[Process {process.pid}]: {out!r}")
            if err:
                log.error(f"[Process {process.pid}]: {err!r}")

            if time.time() - last_access > self.wait_time:
                process.kill()
                break

            time.sleep(self.wait_time * 0.1)

        # Subprocess is terminated; flush output
        out = process.stdout.readline()
        err = process.stderr.readline()
        if out:
            log.info(f"[Process {process.pid}]: {out!r}")
        if err:
            log.error(f"[Process {process.pid}]: {err!r}")

        return process.returncode
