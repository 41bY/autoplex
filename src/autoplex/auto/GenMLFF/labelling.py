"""Jobs to create training data for ML potentials."""

import os
import logging
import subprocess
from glob import glob
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
import numpy as np

from ase import Atoms
from ase.io import read, write
from jobflow import job, Flow, Maker, Response

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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


@dataclass
class QEstaticLabelling(Maker):
    """
    Maker to set up and run Quantum Espresso static calculations for input structures, including bulk, isolated atoms, and dimers.
    Parameters
    ----------
    name: str
        Name of the flow.
    qe_run_cmd: str
        String with the command to run QE (including its executable path/or application name).
    fname_pwi_template: str
        Path to file containing the template computational parameters.
    fname_structures: str
        Path to ASE-readible file containing the structures to be computed.
    num_qe_workers: int | None
        Number of workers to use for the calculations. If None, defaults to the number of structures.
    """

    name: str = "do_qe_labelling"
    qe_run_cmd: str | None = None #String with the command to run QE (including its executable path/or application name)
    fname_pwi_template: str | None = None #Path to file containing the template computational parameters
    fname_structures: str | None = None #Path to ASE-readible file containing the structures to be computed
    num_qe_workers: int | None = None #Number of workers to use for the calculations. 

    def make(self):
        #Define jobs
        joblist = []

        # Load structures
        if self.fname_structures is None: raise ValueError("No structure paths provided. Please provide a list of paths to structures.")
        structures = read(self.fname_structures, index=":")
        if len(structures) == 0: raise ValueError("No structures found in the provided file. Please provide a valid file with structures.")

        # Check pwi template
        pwi_template_lines = self.check_pwi_template(self.fname_pwi_template)

        # Write pwi input files for each structure
        work_dir = os.getcwd()
        path_to_qe_workdir = os.path.join(work_dir, "scf_files")
        os.makedirs(path_to_qe_workdir, exist_ok=True)

        for i, structure in enumerate(structures):
            fname_new_pwi = os.path.join(path_to_qe_workdir, f"structure_{i}.pwi")
            self.write_pwi(
                fname_pwi_output=fname_new_pwi,
                structure=structure, 
                pwi_template=pwi_template_lines, 
                )

        # Set number of QE workers
        if self.num_qe_workers is None: # 1 worker per structure (all DFT jobs in parallel)
            num_qe_workers = len(glob(os.path.join(path_to_qe_workdir, "*.pwi")))
        else: 
            num_qe_workers = self.num_qe_workers

        # Launch QE workers            
        success_of_workers = []
        for id_qe_worker in range(num_qe_workers):
           qe_worker = self.run_qe_worker(
               id=id_qe_worker,
               command=self.qe_run_cmd,
               work_dir=path_to_qe_workdir
               )
           
           qe_worker.name = f"run_qe_worker_{id_qe_worker}"
           joblist.append(qe_worker)
           success_of_workers.append(qe_worker.output)

        # Output is a list of success status, one for each worker
        # The success status is a dictionary with the pwo file name as key and the calculation success status as value (True/False)
        return Response(replace=Flow(joblist), output=success_of_workers)

    def check_pwi_template(self, fname_template):
        """
        Check the pwi template file for the required parameters.
        """
        # Read template file
        tmp_pwi_lines = []
        with open(fname_template, 'r') as f:
            tmp_pwi_lines = f.readlines()

        # Modify lines with structure information: 
        # Assume ntyp, atom_types and pseudoptentials are already defined in the template and consistent with the structures
        # Assume ibrav=0 and Kspacing is already defined in the template
        idx_nat_line, idx_kpoints_line, idx_pos_line, idx_cell_line = 0, 0, 0, 0
        for i, line in enumerate(tmp_pwi_lines):
            if 'nat' in line: idx_nat_line = i
            
            elif 'K_POINTS' in line: idx_kpoints_line = i

            elif 'ATOMIC_POSITIONS' in line: idx_pos_line = i

            elif 'CELL_PARAMETERS' in line: idx_cell_line = i
        
        # Set nat line
        if idx_nat_line == 0: # nat not defined, assume nat = 0
            raise ValueError("Number of atoms line not defined in the template file. Please define \'nat =\' in the template file.")
        else:
            tmp_pwi_lines[idx_nat_line] = f'nat = \n'
        
        # Set K_points lines
        # TODO: Set K_points lines based on the structure and K-spacing
        if idx_kpoints_line == 0: # K_POINTS not defined, assume Gamma point
            kpoints_lines = ["\nK_POINTS Gamma\n"]
        elif idx_kpoints_line > 0:
            kpoints_lines = tmp_pwi_lines[idx_kpoints_line:idx_kpoints_line+1]
            del tmp_pwi_lines[idx_kpoints_line:]

        # Cancel lines with ATOMIC_POSITIONS and CELL_PARAMETERS
        if idx_pos_line == 0 and idx_cell_line > 0:
            idx_to_delete = idx_pos_line
            del(tmp_pwi_lines[idx_to_delete:])
        
        elif idx_pos_line > 0 and idx_cell_line == 0:
            idx_to_delete = idx_pos_line
            del(tmp_pwi_lines[idx_to_delete:])
        
        elif idx_pos_line > 0 and idx_cell_line > 0:
            idx_to_delete = min([idx_pos_line, idx_cell_line])
            del(tmp_pwi_lines[idx_to_delete:])
        
        # Build final template lines
        tmp_pwi_lines = tmp_pwi_lines + kpoints_lines

        return tmp_pwi_lines

    def write_pwi(
            self, 
            fname_pwi_output: str,
            structure: Atoms, 
            pwi_template: list[str],
            ):
        """
        Write the pwi input file for the given structure.
        """
        # Write nat line
        idx_nat_line, nat = 0, len(structure)
        for idx, line in enumerate(pwi_template):
            if 'nat =' in line: idx_nat_line = idx
        pwi_template[idx_nat_line] = f'nat = {nat}\n'

        #Write cell lines
        cell_lines = ["\nCELL_PARAMETERS (angstrom)\n"]
        cell_lines += [f"{structure.cell[i, 0]:.10f} {structure.cell[i, 1]:.10f} {structure.cell[i, 2]:.10f}\n" for i in range(3)]
        
        #Write positions lines
        pos_lines = ["\nATOMIC_POSITIONS (angstrom)\n"]
        for i, atom in enumerate(structure):
            pos_lines.append(f"{atom.symbol} {structure.positions[i, 0]:.10f} {structure.positions[i, 1]:.10f} {structure.positions[i, 2]:.10f}\n")

        # Write the modified lines to the new pwi file
        with open(fname_pwi_output, 'w') as f:
            for line in pwi_template:
                f.write(line)
            for line in cell_lines:
                f.write(line)
            for line in pos_lines:
                f.write(line)

    @job
    def run_qe_worker(
            self, 
            id,
            command,
            work_dir,
            ):
        """
        Run the QE command in a subprocess.
        """
        #Get pwi files
        pwi_files = glob(os.path.join(work_dir, "*.pwi"))

        #Check pwo does not exist
        success_pwo = {}
        for pwi in pwi_files:
            #Try locking the pwi file
            lock_pwi, pwo_fname = self.lock_input(pwi_fname=pwi, worker_id=id)

            if lock_pwi == "": continue #Skip to next pwi if lock failed

            #Launch QE calculation
            success = self.run_qe(command=command, fname_pwi=lock_pwi, fname_pwo=pwo_fname)

            #Set success status
            success_pwo[pwo_fname] = success

        return success_pwo

    def run_qe(self, command, fname_pwi, fname_pwo):
        """
        Run the QE command in a subprocess. Execute one QuantumEspresso calculation on the current input file.
        """
        #Assemble QE command
        run_cmd = f"{command} < {fname_pwi} >> {fname_pwo}"

        success = False
        try:        
            # Launch QE and wait till ending
            subprocess.run(run_cmd, shell=True, check=True, executable="/bin/bash")
            
            success = True
        
        except subprocess.CalledProcessError as e:
            
            success = False

        return success
    
    def lock_input(self, pwi_fname, worker_id):
        
        pwi_lock_fname = ""
        #Check if pwo exists
        pwo_fname = pwi_fname.replace('.pwi', '.pwo')
        if os.path.exists(pwo_fname): return pwi_lock_fname #If exists, skip to next pwi

        # Try to lock the pwi file by renaming it
        pwi_lock_fname = f'{pwi_fname}.lock_{worker_id}'
        try:
            os.rename(f'{pwi_fname}', f'{pwi_lock_fname}')
        except Exception as e:
            pwi_lock_fname = ""  
        
        return pwi_lock_fname, pwo_fname