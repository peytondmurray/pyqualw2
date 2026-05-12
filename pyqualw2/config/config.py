import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Self

from .. import utils
from .inputs import (
    BathymetryInput,
    FlowData,
    FlowInput,
    MetDataInput,
    NoopInput,
    ProfileInput,
    TempDataInput,
    W2ConSimpleInput,
)

DEFAULT_CEQUALW2_EXE_PATH = (
    Path(__file__).parent.parent.parent / "cequalw2" / "w2_v45_64.exe"
)


class Config:
    """A container that holds all configuration data for a simulation."""

    def __init__(
        self,
        con: W2ConSimpleInput,
        bathymetry: BathymetryInput,
        profile: ProfileInput,
        met_data: MetDataInput,
        shade: NoopInput,
        temp_data: list[TempDataInput],
        branch_inflow: list[FlowInput],
        branch_outflow: list[FlowInput],
        branch_evaporation: list[FlowInput],
        temperature_tributaries: list[NoopInput],
        wind_sheltering: NoopInput,
        cequalw2_path: PathLike,
        name: str,
    ):
        self.con = con
        self.bathymetry = bathymetry
        self.profile = profile
        self.met_data = met_data
        self.shade = shade
        self.wind_sheltering = wind_sheltering
        self.temperature_tributaries = temperature_tributaries

        # Branch-specific data
        self.temp_data = temp_data
        self.branch_inflow = branch_inflow
        self.branch_outflow = branch_outflow
        self.branch_evaporation = branch_evaporation

        self.cequalw2_path = Path(cequalw2_path)

        self.name = name

    def parameterize(self, parameters: dict[str, Any]) -> list[Config]:
        """Override settings for each parameter, generating a new set of Configs.

        See the documentation for pytest's `pytest.mark.parametrize` function, which
        was the inspiration for this function.

        Parameters
        ----------
        parameters : dict[str, Any]
            Configuration settings to override

        Returns
        -------
        list["Config"]
            A list of Config objects, one for each value specified in `parameters`
        """
        results = []

        sim_start, _, _ = self.con.timedata

        # Set the metrology data to have the same year as the simulation year
        sim_year = utils.jday_to_date(sim_start).year

        for filename in parameters["met_data"]:
            # Fudge the met data dates so that cequalw2 can simulate scenarios
            # with various metrology data
            met_data = MetDataInput.from_file(filename)
            met_data.set_false_julian_year(sim_year)

            results.append(
                Config(
                    con=self.con,
                    bathymetry=self.bathymetry,
                    profile=self.profile,
                    met_data=met_data,
                    temp_data=self.temp_data,
                    shade=self.shade,
                    branch_inflow=self.branch_inflow,
                    branch_outflow=self.branch_outflow,
                    branch_evaporation=self.branch_evaporation,
                    temperature_tributaries=self.temperature_tributaries,
                    wind_sheltering=self.wind_sheltering,
                    cequalw2_path=self.cequalw2_path,
                    name=f"{self.name}_met{met_data.original_year}",
                )
            )
        return results

    @classmethod
    def from_files(
        cls,
        name: str,
        con: PathLike,
        bathymetry: PathLike,
        profile: PathLike,
        met_data: PathLike,
        shade: PathLike,
        wind_sheltering: PathLike,
        temp_data: list[PathLike],
        temperature_tributaries: list[PathLike],
        cequalw2_path: PathLike = DEFAULT_CEQUALW2_EXE_PATH,
        flow_data: PathLike | None = None,
        flow_data_date_col: str | None = None,
        flow_data_inflow_cols: list[list[str]] | None = None,
        flow_data_outflow_cols: list[list[str]] | None = None,
        flow_data_evaporation_cols: list[list[str]] | None = None,
        branch_inflow: list[PathLike] | None = None,
        branch_outflow: list[PathLike] | None = None,
        branch_evaporation: list[PathLike] | None = None,
    ) -> Self:
        """Generate a Config from w2_con, bathymetry, and intial profile files.

        Parameters
        ----------
        name : str
            Name of the configuration; used for labeling simulation output
        con : PathLike
            Path to a qualw2 configuration file, e.g. w2_con.csv
        bathymetry : PathLike
            Path to the bathymetry file, e.g. mbth_wb1.csv
        profile : PathLike
            Path to the intial profile file, e.g. mvpr1.csv
        met_data : PathLike
            Path to the met data input file, e.g.
            historic_data/met_data/2018_CEQUAL_met_inputs.csv, or inputs/mtin_br1.csv
        temp_data : list[PathLike]
            Path to the branch temp data input file, e.g.
            historic_data/temp_data/SJA_2018_temp.csv
        flow_data : PathLike | None
            Path to the inflow flow data input file,
            historic_data/temp_data/2018_Observed_Flow.csv. If provided,
            flow_data_date_col, flow_data_inflow_cols, flow_data_outflow_cols,
            and flow_data_evaporation_cols must also be provided. If not provided,
            branch_inflow, branch_outflow, and branch_evaporation must be provided.
        shade : PathLike
            Path to the shade file, e.g. inputs/mshade.npt
        temperature_tributaries : PathLike
            Path to the temperature tributaries files, e.g. mtdt_br1.csv
        cequalw2_path : PathLike
            Path to the cequalw2 binary. If None, the default cequalw2 shipped with
            pyqualw2 will be used.
        wind_sheltering : PathLike
            Path to the wind sheltering file, e.g. inputs/mwsc.npt
        flow_data_date_col: str | None
            Column name in the flow data to use for date
        flow_data_inflow_cols: list[str] | None
            Column names in the flow data to use for each branch inflow file
        flow_data_outflow_cols: list[str] | None
            Column names in the flow data to use for each branch outflow file
        flow_data_evaporation_cols: list[str] | None
            Column names in the flow data to use for each branch evaporation file
        branch_inflow: list[PathLike] | None
            Paths to individual branch inflow files (if no flow_data provided)
        branch_outflow: list[PathLike] | None
            Paths to individual branch outflow files (if no flow_data provided)
        branch_evaporation: list[PathLike] | None
            Paths to individual branch evaporation files (if no flow_data provided)

        Returns
        -------
        Self
            A Config instance containing all the information needed to run a simulation

        Raises
        ------
        ValueError
            Raised if
                1. Both flow_data and branch_{inflow,outflow,evaporation} were given
                2. flow_data was specified but column names needed to extract
                   branch-specific flow data were not
                3. Neither flow_data nor branch_{inflow,outflow,evaporation} were given
        """
        if flow_data is None:
            if (
                branch_inflow is None
                or branch_outflow is None
                or branch_evaporation is None
            ):
                raise ValueError(
                    "Either historical branch flow data or individual branch inflow, "
                    "outflow, and evaporation files must be specified."
                )
            inflow = [FlowInput.from_file(f) for f in branch_inflow]
            outflow = [FlowInput.from_file(f) for f in branch_outflow]
            evaporation = [FlowInput.from_file(f) for f in branch_evaporation]
        else:
            if any([branch_evaporation, branch_inflow, branch_outflow]):
                raise ValueError(
                    "Either a flow file or (branch evaporation, branch inflow, "
                    "and branch outflow) files must be specified, not both."
                )

            if (
                flow_data_date_col is None
                or flow_data_inflow_cols is None
                or flow_data_outflow_cols is None
                or flow_data_evaporation_cols is None
            ):
                raise ValueError(
                    "Date, inflow, outflow, and evaporation column names for each "
                    "branch must be provided if a historical flow file is specified."
                )

            flow = FlowData.from_file(
                flow_data,
                date_col=flow_data_date_col,
                inflow_cols=flow_data_inflow_cols,
                outflow_cols=flow_data_outflow_cols,
                evaporation_cols=flow_data_evaporation_cols,
            )

            # American Bureau of Reclamation flow data format means that these all
            # appear in the same historical data file; need to split this for cequalw2
            # to ingest
            inflow = flow.to_inflow_inputs()
            outflow = flow.to_outflow_inputs()
            evaporation = flow.to_evaporation_inputs()

        return cls(
            con=W2ConSimpleInput.from_file(con),
            bathymetry=BathymetryInput.from_file(bathymetry),
            profile=ProfileInput.from_file(profile),
            met_data=MetDataInput.from_file(met_data),
            shade=NoopInput.from_file(shade),
            temperature_tributaries=[
                NoopInput.from_file(f) for f in temperature_tributaries
            ],
            wind_sheltering=NoopInput.from_file(wind_sheltering),
            temp_data=[TempDataInput.from_file(f) for f in temp_data],
            cequalw2_path=cequalw2_path,
            branch_inflow=inflow,
            branch_outflow=outflow,
            branch_evaporation=evaporation,
            name=name,
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
        working_directory: PathLike,
        overwrite: bool = False,
        create_parents: bool = True,
    ):
        """Write the configuration to a directory.

        Currently requires a source directory for the remaining unmodified model files.

        Parameters
        ----------
        working_directory : PathLike
            Path to the directory where the configuration files should be written
        overwrite : bool
            If True, overwrite existing files
        create_parents : bool
            If True, create any necessary parent directories.
        """
        working_directory = Path(working_directory)
        inputs = working_directory / "inputs"

        self.con.to_file(
            working_directory / "w2_con.csv", overwrite=True, create_parents=True
        )

        for fname, obj in zip(
            ["mbth_wb1.csv", "mvpr1.npt", "mmet3.csv", "mshade.npt", "mwsc.npt"],
            [
                self.bathymetry,
                self.profile,
                self.met_data,
                self.shade,
                self.wind_sheltering,
            ],
            strict=True,
        ):
            obj.to_file(inputs / fname, overwrite, create_parents)

        # Write the branch-specific files. Note that not all branches have each file,
        # so we iterate over them separately here
        for i, inflow in enumerate(self.branch_inflow, start=1):
            inflow.to_file(inputs / f"mqin_br{i}.csv", overwrite, create_parents)

        # These _should_ be branch-specific, but isn't. Tracking issue:
        # https://github.com/steelhead-dev/pyqualw2/issues/64
        for i, temp in enumerate(self.temp_data, start=1):
            temp.to_file(inputs / f"mtin_br{i}.csv", overwrite, create_parents)

        # Branch flow and temperature changes (rain and rain temperature)
        for i, evaporation in enumerate(self.branch_evaporation, start=1):
            evaporation.to_file(inputs / f"mqdt_br{i}.csv", overwrite, create_parents)
        for i, temperature in enumerate(self.temperature_tributaries, start=1):
            temperature.to_file(inputs / f"mtdt_br{i}.npt", overwrite, create_parents)

        for i, outflow in enumerate(self.branch_outflow, start=1):
            outflow.to_file(inputs / f"mqot_br{i}.csv", overwrite, create_parents)

        # Write the cequalw2 binary to the directory, since it needs to be in the same
        # directory as the data to work (paths are hardcoded in fortran...)
        shutil.copy(
            self.cequalw2_path.resolve(), working_directory / self.cequalw2_path.name
        )
