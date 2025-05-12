"""Jobs to create training data for ML potentials."""

import logging
import os
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
import numpy as np

from ase import Atoms
from ase.io import read, write
from jobflow import Maker

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
    mlip_type: str | None = None
    mlip_kwargs: dict | None = None
    structure_paths: list[str] | None = None
    structure_types: list[str] | None = None
    isolated_atom: bool = False
    isolated_species: list[str] | None = None
    isolatedatom_box: list[float] = field(default_factory=lambda: [20, 20, 20])
    dimer: bool = False
    dimer_pairs: list[tuple[str]] | None = None  
    dimer_range: list[float] | None = None
    dimer_num: int = 21
    dimer_box: list[float] = field(default_factory=lambda: [20, 20, 20])


    def make(self):
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

        # Check if structure paths are correctly provided
        if self.structure_paths is None:
            raise ValueError("No structure paths provided. Please provide a list of paths to structures.")
        if self.structure_types is not None:
            assert len(self.structure_paths) == len(self.structure_types), "The length of structure_paths and structure_types must be the same."
        else: #Assume structure types are all bulk
            self.structure_types = ["bulk"] * len(self.structure_paths)        

        #Define ase atoms with results
        computed_structures = []

        # Perform DFT-batch calculations
        for structures_type, structures_path in zip(self.structure_types, self.structure_paths):
            # Load structures
            ase_structures = read(structures_path, index=":")

            # Perform batch static calculations
            tmp_structures = self.compute_structures(
                atoms=ase_structures,
                ase_calculator=mlff_ase_calc,
            )
            #Update structure type info
            for structure in tmp_structures:
                structure.info["structure_type"] = structures_type
            
            # Save computed structures
            computed_structures += tmp_structures
        
        #Perform dimer calculations based on atoms
        if self.dimer:
            
            #Get list of pairs
            if self.dimer_pairs is None:
                #Get unique atomic species
                unique_species = set()
                for atoms in computed_structures:
                    unique_species.update(atoms.get_chemical_symbols())
                unique_species = list(unique_species)
                logging.info(f"Unique species found in structures: {unique_species}")

                #Get every combination of two species
                dimer_pairs = list(combinations_with_replacement(unique_species, 2))
                num_pairs = len(dimer_pairs)
                logging.info(f"Number of dimer pairs: {num_pairs}")
            else:
                dimer_pairs = self.dimer_pairs

            #Create dimers geometries (ASE atoms objects)
            dimers = self.create_dimers(
                pairs=dimer_pairs,
                dimer_range=self.dimer_range,
                dimer_num=self.dimer_num,
                dimer_box=self.dimer_box,
            )

            #Perform batch static calculations
            tmp_dimers = self.compute_structures(
                atoms=dimers,
                ase_calculator=mlff_ase_calc,
            )

            #Update structure type info
            for dimer in tmp_dimers:
                dimer.info["structure_type"] = "dimer"

            # Save computed structures
            computed_structures += tmp_dimers            

        #Isolated atoms with MLIP can be problematic...
        if self.isolated_atom: 
            logging.warning("Isolated atom calculations with MLIP can be problematic: provide isolated atom energies instead.")
            pass

        #Define output directory
        output_dir = os.path.join(os.getcwd(), "batch_results")       
        os.makedirs(output_dir, exist_ok=True) 

        # Save structures to file
        output_file = os.path.join(output_dir, f"labels.extxyz")
        write(output_file, computed_structures, format="extxyz")
        logging.info(f"Saved structures to {output_file}")
        
        # Return the paths to the computed structures
        return output_file

    def compute_structures(
        self, 
        atoms: list[Atoms] | None = None,
        ase_calculator: object | None = None,
        ):
        """
        Compute structures contained in ASE atoms object using the given calculator.

        Parameters
        ----------
        atoms : list[Atoms]
            List of ASE Atoms objects.
        ase_calculator : object | None
            ASE calculator object to use for computations. If None, defaults to the
            calculator set in the class.

        Returns
        -------
        list[Atoms]
            List of ASE Atoms objects representing the computed structures.
        """
        if atoms is None:
            raise ValueError("Atoms list cannot be None.")
        if ase_calculator is None:
            raise ValueError("ASE calculator cannot be None.")
        
        #Perform batch static calculations
        for frame in atoms:
            # Set calculator
            frame.calc = ase_calculator
            
            # Get energy, forces and stress updating atoms object
            energy = frame.get_potential_energy()    
            forces = frame.get_forces()
            stress = frame.get_stress()

            # Save energy, forces and stress to atoms object
            frame.info["REF_energy"] = energy
            frame.arrays["REF_forces"] = forces
            frame.info["REF_stress"] = stress

            # Disconnect calculator to avoid io proble
            frame.calc = None
        
        return atoms

    def create_dimers(
        self,
        pairs: list[tuple[str]] | None = None,
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
            List of ASE Atoms objects representing the created dimers.
        """
        if dimer_range is None:
            dimer_range = [0.5, 2.0]
        if dimer_box is None:
            dimer_box = [20, 20, 20]

        dimer_atoms = []
        dimer_distances = np.linspace(dimer_range[0], dimer_range[1], dimer_num)
        for pair in pairs:
            for distance in dimer_distances:
                # Create dimer structure as ase atoms
                atoms = Atoms(
                    symbols=pair, 
                    positions=[[0.0, 0.0, 0.0], 
                    [distance, 0.0, 0.0]], 
                    cell=dimer_box, 
                    pbc=True)

                #Append dimer atoms to list
                dimer_atoms.append(atoms)

        return dimer_atoms

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
