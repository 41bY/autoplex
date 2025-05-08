"""Jobs to create training data for ML potentials."""

import logging
import os
import traceback
from dataclasses import dataclass, field
import numpy as np

from ase import Atoms
from ase.io import read, write
from jobflow import Maker, job, Response

from autoplex.data.common.utils import ElementCollection

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


#My methods
@dataclass
class MLIPStaticLabelling(Maker):
    """
    Maker to set up and run MLIP static calculations for input structures, including bulk, isolated atoms, and dimers.

    Parameters
    ----------
    name: str
        Name of the flow.
    isolated_atom: bool
        If true, perform single-point calculations for isolated atoms. Default is False.
    isolated_species: list[str]
        List of species for which to perform isolated atom calculations. If None,
        species will be automatically derived from the 'structures' list. Default is None.
    isolatedatom_box: list[float]
        List of the lattice constants for a isolated_atom configuration.
    dimer: bool
        If true, perform single-point calculations for dimers. Default is False.
    dimer_box: list[float]
        The lattice constants of a dimer box.
    dimer_species: list[str]
        List of species for which to perform dimer calculations. If None, species
        will be derived from the 'structures' list. Default is None.
    dimer_range: list[float]
        Range of distances for dimer calculations.
    dimer_num: int
        Number of different distances to consider for dimer calculations.
    mlip_type: str
        Type of MLIP calculator to use. Currently available: 'MACE'. Default is None.
    mlip_path: str
        Path to the MLIP-model that will be loaded as ase calculator object.        
    mlip_kwargs: dict
        Dictionary of custom parameters for mlip ase calculator.

    Returns
    -------
    dict
        A dictionary containing:
        - 'dirs_of_output': List of directories containing output mlip data.
        - 'config_type': List of configuration types corresponding to each directory.
    """

    name: str = "do_mlip_labelling"
    isolated_atom: bool = False
    isolated_species: list[str] | None = None
    isolatedatom_box: list[float] = field(default_factory=lambda: [20, 20, 20])
    dimer: bool = False
    dimer_box: list[float] = field(default_factory=lambda: [20, 20, 20])
    dimer_species: list[str] | None = None
    dimer_range: list[float] | None = None
    dimer_num: int = 21
    mlip_type: str | None = None
    mlip_kwargs: dict | None = None

    @job
    def make(
        self,
        structure_paths: list[str], 
        config_type: str | None = None,
        ):
        """
        Maker to set up and run MLIP static batch calculations.

        Parameters
        ----------
        structures : list[str]
            List of path containing ase-readibile files of structures for which to run the static MLIP labelling. If None,
            no bulk calculations will be performed. Default is None.
        config_type : str
            Configuration types corresponding to the structures. If None, defaults
            to 'bulk'. Default is None.
        """
        # Load MLIP calculator
        mlff_ase_calc = self.load_mlff_model(
            mlip_type=self.mlip_type,
            mlip_kwargs=self.mlip_kwargs,
        )

        # Define output dictionary
        dirs: dict[str, list[str]] = {"dirs_of_output": [], "config_type": []}    

        # Load ASE atoms objects
        frames = []
        for structure in structure_paths:
            frames += read(structure, index=":")

        #Define output directory
        output_batch_dir = os.path.join(os.getcwd(), "batch_results")       
        os.makedirs(output_batch_dir, exist_ok=True) 
        
        #Perform batch static calculations
        for idx, frame in enumerate(frames):
            # Set calculator
            frame.calc = calc
            # Get energy, forces and stress updating atoms object
            frame.get_potential_energy()    
            
            #Set output file
            output_file = os.path.join(output_batch_dir, f"result_{idx}.xyz")
            # Write results to file
            write(output_file, frame)

            # Append output file to output dirs
            dirs["dirs_of_output"].append(output_file)
            if config_type:
                dirs["config_type"].append(config_type)
            else:
                dirs["config_type"].append("bulk")

        if self.isolated_atom: #Isolated atoms with MLIP can be problematic...
            pass

        if self.dimer:
            #Define output directory
            output_dimer_dir = os.path.join(os.getcwd(), "dimer_results")   

            try:
                atoms = [at for at in frames]
                if self.dimer_species is not None:
                    dimer_syms = self.dimer_species
                elif (self.dimer_species is None) and (structure_paths is not None):
                    # Get the species from the database
                    dimer_syms = ElementCollection(atoms).get_species()
                pairs_list = ElementCollection(atoms).find_element_pairs(dimer_syms)
                for pair in pairs_list:
                    for dimer_i in range(self.dimer_num):
                        if self.dimer_range is not None:
                            dimer_distance = self.dimer_range[0] + (
                                self.dimer_range[1] - self.dimer_range[0]
                            ) * float(dimer_i) / float(
                                self.dimer_num - 1 + 0.000000000001
                            )

                        #Build dimer structure as ase atoms
                        dimer_atoms = Atoms(
                            symbols=[pair[0], pair[1]],
                            positions=[[0.0, 0.0, 0.0], [dimer_distance, 0.0, 0.0]],
                            cell=self.dimer_box,
                            pbc=True,
                            )

                        # Set calculator
                        dimer_atoms.calc = calc
                        # Get energy, forces and stress updating atoms object
                        dimer_atoms.get_potential_energy()    
                        
                        #Set output file
                        output_file = os.path.join(output_dimer_dir, f"dimer_{pair[0]}{pair[1]}_{dimer_distance}.xyz")
                        # Write results to file
                        write(output_file, frame)

                        # Append output file to output dirs
                        dirs["dirs_of_output"].append(output_file)
                        dirs["config_type"].append("dimer")

            except ValueError:
                logging.error("Unknown atom types in dimers!")
                traceback.print_exc()
        
    def load_mlff_model(
            self, 
            mlip_type: str = None,
            mlip_kwargs: dict = None,
            ):
        
        if mlip_type == 'MACE':
            try:
                from mace.calculators import MACECalculator
                ase_calc = MACECalculator(**mlip_kwargs)
            except ImportError:
                logging.error("Cannot use MACE model, MACE is not installed. Please install mace-torch package to use this feature.")
                raise 
        
        elif mlip_type == 'GRACE':
            try:
                from tensorpotential.calculator import TPCalculator
                ase_calc = TPCalculator(**mlip_kwargs)
            except ImportError:
                logging.error("Cannot use GRACE model, GRACE is not installed. Please install tensorpotential package to use this feature.")
                raise
        
        elif mlip_type == 'DP':
            try:
                from deepmd.calculator import DP
                ase_calc = DP(**mlip_kwargs)
            except ImportError:
                logging.error("Cannot use DP model, DP is not installed. Please install deepmd package to use this feature.")
                raise
        
        else:
            raise ValueError(f"Unknown forcefield type: {mlip_type}\n\
                             Available forcefield types: MACE, GRACE, DP")

        return ase_calc

    def compute_dimers(
        self,
        atoms: list[Atoms] | None = None,
        dimer_range: list[float] | None = None,
        dimer_num: int = 21,
        dimer_box: list[float] | None = None,
    ):
        """
        Compute dimers for the given atoms.

        Parameters
        ----------
        atoms : list[Atoms]
            List of ASE Atoms objects.
        dimer_range : list[float] | None
            Range of distances for dimer calculations. If None, defaults to [0.5, 2.0].
        dimer_num : int
            Number of different distances to consider for dimer calculations.
        dimer_box : list[float] | None
            The lattice constants of a dimer box. If None, defaults to [20, 20, 20].

        Returns
        -------
        list[Atoms]
            List of ASE Atoms objects representing the computed dimers.
        """
        if dimer_range is None:
            dimer_range = [0.5, 2.0]
        if dimer_box is None:
            dimer_box = [20, 20, 20]

        dimer_atoms = []
        for atom in atoms:
            dimer_distances = np.linspace(dimer_range[0], dimer_range[1], dimer_num)


            for i in range(dimer_num):
                dimer_distance = dimer_range[0] + (
                    dimer_range[1] - dimer_range[0]
                ) * float(i) / float(dimer_num - 1 + 0.000000000001)
                dimer_atoms.append(
                    Atoms(
                        symbols=atom.symbols,
                        positions=[
                            [0.0, 0.0, 0.0],
                            [dimer_distance, 0.0, 0.0],
                        ],
                        cell=dimer_box,
                        pbc=True,
                    )
                )
        return dimer_atoms