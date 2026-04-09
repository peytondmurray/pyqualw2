import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Self

from .inputs import (
    BathymetryInput,
    FlowDataInput,
    MetDataInput,
    ProfileInput,
    TempDataInput,
    W2ConSimpleInput,
)


class Config:
    """A container that holds all configuration data for a simulation."""

    def __init__(
        self,
        con: W2ConSimpleInput,
        bathymetry: BathymetryInput,
        profile: ProfileInput,
        temp_data: TempDataInput,
        met_data: MetDataInput,
        flow_data: FlowDataInput,
    ):
        self.con = con
        self.bathymetry = bathymetry
        self.profile = profile
        self.temp_data = temp_data
        self.met_data = met_data
        self.flow_data = flow_data

    def parameterize(self, parameters: dict[str, Any]) -> list["Config"]:
        """Parameterize the config by creating a new config for each met data file.

        Parameters
        ----------
        parameters : dict[str, Any]
            A dictionary of parameters to parameterize the configuration with.
        """
        results = []

        sim_start, _sim_end, _sim_year = self.con.timedata

        for filename in parameters["met_data"]:
            met_data = MetDataInput.from_file(filename)
            # Fudge the met data dates so that qualw2 runs with arbitrary years
            met_data.set_false_julian_day(sim_start)
            results.append(
                Config(
                    self.con,
                    self.bathymetry,
                    self.profile,
                    self.temp_data,
                    met_data,
                    self.flow_data,
                )
            )
        return results

    @classmethod
    def from_files(
        cls,
        con: PathLike,
        bathymetry: PathLike,
        profile: PathLike,
        temp_data: PathLike,
        met_data: PathLike,
        flow_data: PathLike,
    ) -> Self:
        """Generate a Config from w2_con, bathymetry, and intial profile files.

        Parameters
        ----------
        con : PathLike
            Path to a qualw2 configuration file, e.g. w2_con.csv
        bathymetry : PathLike
            Path to the bathymetry file, e.g. mbth_wb1.csv
        profile : PathLike
            Path to the intial profile file, e.g. mvpr1.csv
        met_data : PathLike
            Path to the met data input file, e.g. mmet3.csv
        temp_data : PathLike
            Path to the infow temp data input file, e.g. mmet3.csv
        flow_data : PathLike
            Path to the inflow flow data input file, e.g. mqin_br.csv.

        Returns
        -------
        Self
            A Config instance containing all the information needed to run a simulation
        """
        return cls(
            con=W2ConSimpleInput.from_file(con),
            bathymetry=BathymetryInput.from_file(bathymetry),
            profile=ProfileInput.from_file(profile),
            temp_data=TempDataInput.from_file(temp_data),
            met_data=MetDataInput.from_file(met_data),
            flow_data=FlowDataInput.from_file(flow_data),
        )

    @classmethod
    def from_settings(cls, settings: PathLike) -> Self:
        """Generate a Config from a settings.toml file.

        Parameters
        ----------
        settings : PathLike
            Path to a settings.toml file

        Returns
        -------
        Self
            A config instance containing all the information needed to run a simulation

        """
        raise NotImplementedError

    def to_directory(
        self,
        source_directory: PathLike,
        working_directory: PathLike,
        overwrite: bool = False,
        create_parents: bool = True,
    ):
        """Write the configuration to a directory.

        Currently requires a source directory for the remaining unmodified model files.

        Parameters
        ----------
        source_directory : PathLike
            Path to the directory where the model files are read from
        working_directory : PathLike
            Path to the directory where the configuration files should be written
        overwrite : bool
            If True, overwrite existing files
        create_parents : bool
            If True, create any necessary parent directories.
        """
        working_path = Path(working_directory)
        inputs_dir_path = working_path / "inputs"

        # Copy the source directory to the working directory
        shutil.copytree(source_directory, working_path, dirs_exist_ok=True)

        self.con.to_file(
            working_path / "w2_con.csv", overwrite=True, create_parents=True
        )

        fnames = ["mbth_wb1.csv", "mvpr1.npt"]

        for fname, obj in zip(
            fnames,
            [self.bathymetry, self.profile],
            strict=True,
        ):
            obj.to_file(inputs_dir_path / fname, overwrite, create_parents)

        self.met_data.to_file(
            inputs_dir_path / "mmet3.csv", overwrite=True, create_parents=True
        )

        for i in range(1, self.con.branchdata + 1):
            if i == 1:
                self.flow_data.to_file(
                    inputs_dir_path / ("mqin_br" + str(i) + ".csv"),
                    overwrite=True,
                    create_parents=True,
                )
                self.flow_data.to_evap_csv_file(
                    inputs_dir_path / ("mqdt_br" + str(i) + ".csv"),
                    overwrite=True,
                    create_parents=True,
                )
                self.flow_data.to_outflow_csv_file(
                    inputs_dir_path / ("mqot_br" + str(i) + ".csv"),
                    overwrite=True,
                    create_parents=True,
                )
                self.temp_data.to_file(
                    inputs_dir_path / ("mtin_br" + str(i) + ".csv"),
                    overwrite=True,
                    create_parents=True,
                )
            else:
                self.flow_data.to_file(
                    inputs_dir_path / ("mqin_br" + str(i) + ".csv"),
                    overwrite=True,
                    create_parents=True,
                    zero_flow=True,
                )
                self.temp_data.to_file(
                    inputs_dir_path / ("mtin_br" + str(i) + ".csv"),
                    overwrite=True,
                    create_parents=True,
                )
