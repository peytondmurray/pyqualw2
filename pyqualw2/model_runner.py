import logging
import os
import platform
import subprocess
import tempfile
import time
from os import PathLike
from pathlib import Path

from pyqualw2.config.config import Config
from pyqualw2.utils import is_notebook

log = logging.getLogger(__name__)


class ModelRunner:
    """A class for exceuting the Qualw2 model.

    Parameters
    ----------
    configuration : Config | list[Config]
        The configuration to run the model with.
    output_dir : Path
        The directory to save the output files to.
    wait_time : int, optional
        The time to wait for the model to finish running, by default 30 seconds.
    """

    def __init__(
        self,
        configuration: Config | list[Config],
        output_dir: PathLike | str,
        wait_time: int = 10,
    ):
        if isinstance(configuration, Config):
            self.configs = [configuration]
        else:
            self.configs = configuration

        self.output_dir = Path(output_dir)
        self.wait_time = wait_time

    def run(self, overwrite: bool = False):
        """Run the model for each configuration in the list.

        Parameters
        ----------
        overwrite : bool
            If True, overwrite the output directory; if False and the output directory
            already exists, an error will be thrown

        Raises
        ------
        ValueError
            Raised if the cequalw2 subprocess returned nonzero
        """
        for config in self.configs:
            with tempfile.TemporaryDirectory() as tempdir:
                wd = Path(tempdir)

                # Make the outputs directory, otherwise cequalw2 will fail
                (wd / "outputs").mkdir(exist_ok=True)
                config.to_directory(wd, overwrite=True)

                retcode = self.run_model(wd)
                if retcode is None or retcode == 0:
                    # The subprocess return code is None if it was terminated early
                    # (i.e. the window was still open, as it always is when the
                    # simulation completes)
                    self.save_outputs(
                        wd,
                        self.output_dir / config.name,
                        overwrite=overwrite,
                    )
                else:
                    raise ValueError(
                        f"cequalw2 subprocess returned error code {retcode}"
                    )

    def save_outputs(
        self, wd: Path, output_dir: PathLike | str, overwrite: bool = False
    ):
        """Copy the cequalw2-generated outputs to the output directory.

        We don't use `Path.copy()` here because

        - We need to check whether the output_dir contains any files that would be
          overwritten
        - We could just do `shutil.rmtree(output_dir)` to ensure no files exist where we
          would like to copy the simulation output to, but that would delete e.g. your
          home directory if you pass $HOME as the output_dir

        Instead, get the list of files to copy over. Check if any exist and raise an
        error if the user has not passed `overwrite=True`. Then proceed to copy each
        file, deleting any existing file if any exists and `overwrite=True`

        Parameters
        ----------
        wd : Path
            Working directory where cequalw2 is run
        output_dir : PathLike | str
            Directory where the user has asked for results to be placed
        overwrite : bool
            If True, overwrite the output directory; if False and the output directory
            already exists, an error will be thrown

        Raises
        ------
        FileExistsError
            Raised if `overwrite=False` is passed and the `output_dir` already contains
            files that would be overwritten by simulation output
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        copy_from = [f for f in wd.glob("**/*") if f.is_file()]
        copy_to = [output_dir / f.relative_to(wd) for f in copy_from]

        if not overwrite and any([f.exists() for f in copy_to]):
            raise FileExistsError(
                f"Cannot save output to {output_dir}: files would be overwritten. "
                "To overwrite existing output, call `ModelRunner.run(overwrite=True)`."
            )

        for f_from, f_to in zip(copy_from, copy_to, strict=True):
            if not f_to.parent.exists():
                f_to.parent.mkdir(parents=True, exist_ok=True)

            if f_to.exists() and overwrite:
                f_to.unlink()

            f_from.copy(f_to)

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

        # Flag to indicate if the simulation is idle and should be killed
        # needed for windows to handle clean-up of tmp directories
        idle_kill_flag = False

        last_access = time.time()

        stdout = None
        stderr = None
        if not is_notebook():
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE

        process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            cwd=str(wd),
        )

        if process.stdout is not None and process.stderr is not None:
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
            if process.stdout is not None and process.stderr is not None:
                out = process.stdout.readline()
                err = process.stderr.readline()
                if out:
                    log.info(f"[Process {process.pid}]: {out!r}")
                if err:
                    log.error(f"[Process {process.pid}]: {err!r}")

            if time.time() - last_access > self.wait_time:
                process.kill()
                idle_kill_flag = True
                break

            time.sleep(self.wait_time * 0.1)

        if platform.system() == "Windows":
            process.wait()

        if process.stdout is not None and process.stderr is not None:
            # Subprocess is terminated; flush output
            out = process.stdout.readline()
            err = process.stderr.readline()
            if out:
                log.info(f"[Process {process.pid}]: {out!r}")
            if err:
                log.error(f"[Process {process.pid}]: {err!r}")

        if idle_kill_flag:
            return 0
        else:
            return process.returncode
