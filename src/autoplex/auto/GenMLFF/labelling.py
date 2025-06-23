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
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import job, Flow, Maker, Response


from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.powerups import (
    update_user_incar_settings,
    update_user_potcar_settings,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class VASPStaticLabelling(Maker):
    """
    Maker to set up and run VASP static calculations for input structures.

    It supports custom VASP input parameters and error handlers.

    Parameters
    ----------
    name: str
        Name of the flow.
    fname_structures: str | list[str] | None
        Path or list of paths to ASE-readable files containing the structures to be computed.
    custom_incar: dict
        Dictionary of custom VASP input parameters. If provided, will update the
        default parameters. Default is None.
    custom_potcar: dict
        Dictionary of POTCAR settings to update. Keys are element symbols, values are the desired POTCAR labels.
        Default is None.

    Returns
    -------
    list[dict]
        A dictionary containing:
        -   'successes' : bool, 'OUTCAR' : str, 'OUTDIR' : str}
    """

    name: str = "do_vasp_labelling"
    fname_structures: str | list[str] | None = None #Path or list[Path] to ASE-readible file containing the structures to be computed
    custom_incar: dict | None = None
    custom_potcar: dict | None = None

    @job
    def make(self):
        """
        Maker to set up and run VASP static calculations.

        Returns
        -------
        """
        #Define list of jobs
        job_list = []

        #Get default VASP settings
        default_custom_set = {
            "ADDGRID": "True",
            "ENCUT": 520,
            "EDIFF": 1e-06,
            "ISMEAR": 0,
            "SIGMA": 0.01,
            "PREC": "Accurate",
            "ISYM": None,
            "KSPACING": 0.2,
            "NPAR": 8,
            "LWAVE": "False",
            "LCHARG": "False",
            "ENAUG": None,
            "GGA": None,
            "ISPIN": None,
            "LAECHG": None,
            "LELF": None,
            "LORBIT": None,
            "LVTOT": None,
            "NSW": None,
            "SYMPREC": None,
            "NELM": 100,
            "LMAXMIX": None,
            "LASPH": None,
            "AMIN": None,
        }

        # Update default settings with custom incar settings
        if self.custom_incar is not None:
            default_custom_set.update(self.custom_incar)
        custom_set = default_custom_set

        #Instantiate the VASP static maker
        st_m = StaticMaker(
            input_set_generator=StaticSetGenerator(user_incar_settings=custom_set),
            run_vasp_kwargs={"handlers": ()},
        )

        #Update the VASP static maker with custom POTCAR settings
        if self.custom_potcar is not None:
            st_m = update_user_potcar_settings(st_m, potcar_updates=self.custom_potcar)

        #Load structures
        structures = self.load_structures(fname_structures=self.fname_structures)

        if structures: #Define 1 VASP job for each structure
            #Get adaptor to convert ASE Atoms to pymatgen Structure
            ase_pmg_adaptor = AseAtomsAdaptor()
            #Define worker outputs
            worker_output = {'success' : [], 'output' : [], 'outdir' : []}
            for idx, struct in enumerate(structures):
                #Convert ASE Atoms object to pymatgen Structure object
                pmg_struct = ase_pmg_adaptor.get_structure(struct)

                #Create static VASP job
                static_job = st_m.make(structure=pmg_struct)
                static_job.name = f"static_bulk_{idx}"

                #Get output
                if static_job.output.state.SUCCESS: success = True
                else: success = False
                outdir = static_job.output.dir_name
                output_fname = outdir +"/OUTCAR"

                #Save output
                worker_output['success'].append(success)
                worker_output['output'].append(output_fname)
                worker_output['outdir'].append(outdir)                

                #Append to the job list
                job_list.append(static_job)
            
            #Define flow
            vasp_flow = Flow(jobs=job_list, output=worker_output, name="vasp_static_labelling")

        else:
            raise ValueError("No structures found to compute with VASP. Exiting.")

        return Response(replace=vasp_flow, output=vasp_flow.output)

    def load_structures(self,
            fname_structures: str | list[str] | None = None,
            ):
        """
        Load structures from a file or a list of files.
        Parameters
        ----------
        fname_structures : str | list[str] | None
            Path or list of paths to ASE-readable files containing the structures to be loaded.
            If None, no structures will be loaded.
        Returns
        -------
        list[Atoms]
            List of ASE Atoms objects representing the loaded structures.
        """
        #Convert fname_structures to a list if it is a string
        if isinstance(fname_structures, str):
            fname_structures = [fname_structures]
        elif fname_structures is None:
            return []
        elif not isinstance(fname_structures, list):
            raise ValueError("fname_structures must be a string or a list of strings.")
        
        #Loop over provided files and load structures
        structures = []
        for fname in fname_structures:
            #Check if all files exist
            if not os.path.exists(fname): raise FileNotFoundError(f"File {fname} does not exist.")
        
            #Read structures from file
            try:
                structures += read(fname, index=":")
            except Exception as e:
                logging.error(f"Error reading file {fname}: {e}")

        return structures

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

def qe_params_from_config(config: dict):
    """
    Return QEstaticLabelling params from a configuration dictionary.

    Args:
        config (dict): Keys should match __init__ parameters. For example:
            {
                "qe_run_cmd": qe_run_cmd,
                "num_qe_workers": num_qe_workers,
                "fname_pwi_template": fname_pwi_template,
                "kspace_resolution" : Kspace_resolution,
                "koffset": Koffset,
                "fname_structures": fname_structures,
            }

    Returns:
        params: dict
            Dictionary with parameters for QEstaticLabelling.
    """
    #Get default parameters
    params = {
        "qe_run_cmd": "pw.x",
        "num_qe_workers": 1,
        "fname_pwi_template": None,
        "kspace_resolution" : None,
        "koffset": [False, False, False],
        "fname_structures": None,
    }    

    # Update parameters with values from the config file
    if config is None: raise ValueError("Configuration file is empty or not properly formatted.")
    params.update(config)

    #Check a valid reference pwi path is provided
    if not os.path.exists(params["fname_pwi_template"]): raise ValueError(f"Reference QE input file '{params['fname_pwi_template']}' not found.")

    return params

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
    fname_structures: str | list[str] | None = None #Path or list[Path] to ASE-readible file containing the structures to be computed
    num_qe_workers: int | None = None #Number of workers to use for the calculations. 
    kspace_resolution: float | None = None #K-space resolution in Angstrom^-1, used to set the K-points in the pwi file
    koffset: list[bool] = field(default_factory=lambda: [False, False, False]) #K-points offset in the pwi file
    
    def make(self):
        #Define jobs
        joblist = []

        # Load structures
        structures = self.load_structures(fname_structures=self.fname_structures)
        if len(structures) == 0:
            logging.info("No structures found to compute with DFT. Exiting.")
            return Response(replace=None, output=[])

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
        outputs = []
        for id_qe_worker in range(num_qe_workers):
           qe_worker = self.run_qe_worker(
               id=id_qe_worker,
               command=self.qe_run_cmd,
               work_dir=path_to_qe_workdir
               )
           
           qe_worker.name = f"run_qe_worker_{id_qe_worker}"
           joblist.append(qe_worker)
           outputs.append(qe_worker.output) #Contains list of dict{'successes', 'pwo_files', 'outdirs'} for each worker

        qe_wrk_flow = Flow(jobs=joblist, output=outputs, name="qe_workers")

        # Output is a list of success status, one for each worker
        # The success status is a dictionary with the pwo file name as key and the calculation success status as value (True/False)
        return Response(replace=qe_wrk_flow, output=qe_wrk_flow.output)

    def load_structures(self,
            fname_structures: str | list[str] | None = None,
            ):
        """
        Load structures from a file or a list of files.
        Parameters
        ----------
        fname_structures : str | list[str] | None
            Path or list of paths to ASE-readable files containing the structures to be loaded.
            If None, no structures will be loaded.
        Returns
        -------
        list[Atoms]
            List of ASE Atoms objects representing the loaded structures.
        """
        #Convert fname_structures to a list if it is a string
        if isinstance(fname_structures, str):
            fname_structures = [fname_structures]
        elif fname_structures is None:
            return []
        elif not isinstance(fname_structures, list):
            raise ValueError("fname_structures must be a string or a list of strings.")
        
        #Loop over provided files and load structures
        structures = []
        for fname in fname_structures:
            #Check if all files exist
            if not os.path.exists(fname): raise FileNotFoundError(f"File {fname} does not exist.")
        
            #Read structures from file
            try:
                structures += read(fname, index=":")
            except Exception as e:
                logging.error(f"Error reading file {fname}: {e}")

        return structures

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
        idx_nat_line, idx_pos_line, idx_cell_line = 0, 0, 0
        for i, line in enumerate(tmp_pwi_lines):
            if 'nat' in line: idx_nat_line = i

            elif 'ATOMIC_POSITIONS' in line: idx_pos_line = i

            elif 'CELL_PARAMETERS' in line: idx_cell_line = i
        
        # Set nat line
        if idx_nat_line == 0: # nat not defined, assume nat = 0
            raise ValueError("Number of atoms line not defined in the template file. Please define \'nat =\' in the template file.")
        else:
            tmp_pwi_lines[idx_nat_line] = f'nat = \n'

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
        # Check pwi lines
        idx_diskio, idx_outdir, idx_nat_line, idx_kpoints_line, nat = 0, 0, 0, 0, len(structure)
        for idx, line in enumerate(pwi_template):
            if 'nat =' in line: idx_nat_line = idx
            elif 'disk_io' in line: idx_diskio = idx
            elif 'outdir' in line: idx_outdir = idx
            elif 'K_POINTS' in line: idx_kpoints_line = i
        
        #Update number of atoms
        pwi_template[idx_nat_line] = f'nat = {nat}\n'

        #Get identifier for this structure
        structure_id = fname_pwi_output.split('/')[-1].replace('.pwi', '')

        #Update outdir based on disk_io
        if idx_diskio == 0 or 'none' not in pwi_template[idx_diskio]: #disk_io is not 'none' (QE default is low for scf)
            if idx_outdir == 0: # outdir not defined, define it
                pwi_template.insert(idx_diskio + 1, f"outdir = {structure_id}\n")
            else: # outdir is defined, update it
                pwi_template[idx_outdir] = f"outdir = '{structure_id}'\n"
        else: #disk_io is 'none', remove outdir line
            if idx_outdir == 0:
                pwi_template.insert(idx_diskio + 1, f"outdir = 'OUT'\n")

        kpoints_lines = self.set_Kpoints(
            tmp_pwi_lines=pwi_template, 
            idx_kpoints_line=idx_kpoints_line, 
            atoms=structure,
            Kspace_resolution=self.kspace_resolution,
            Koffset=self.koffset,
        )        

        #Write cell lines
        cell_lines = ["\nCELL_PARAMETERS (angstrom)\n"]
        cell_lines += [f"{structure.cell[i, 0]:.10f} {structure.cell[i, 1]:.10f} {structure.cell[i, 2]:.10f}\n" for i in range(3)]
        
        #Write positions lines
        pos_lines = ["\nATOMIC_POSITIONS (angstrom)\n"]
        for i, atom in enumerate(structure):
            pos_lines.append(f"{atom.symbol} {structure.positions[i, 0]:.10f} {structure.positions[i, 1]:.10f} {structure.positions[i, 2]:.10f}\n")

        # Write the modified lines to the new pwi file
        with open(fname_pwi_output, 'w') as f:
            for line in pwi_template: #Write reference pwi lines (computational parameters)
                f.write(line)
            for line in kpoints_lines: #Write K-points lines
                f.write(line)
            for line in cell_lines: #Write cell lines
                f.write(line)
            for line in pos_lines: #Write positions lines
                f.write(line)

    def set_Kpoints(self,
            tmp_pwi_lines: list[str],
            idx_kpoints_line: int,
            atoms: Atoms,
            Kspace_resolution: float | None = None,
            Koffset: list[bool] = [False, False, False],
        ):
            """
            Set the K-points in the pwi file based on user definition or K-space resolution.
            """
            # Define K-points lines
            kpoints_lines = []
            
            # K_POINTS line not found
            if idx_kpoints_line == 0:
                if Kspace_resolution is None: # K_POINTS line not found and Kspace_resolution is not defined
                    raise ValueError("K_POINTS line not found in the template file. Please define K_POINTS in the template file or provide Kspace_resolution.")
                else: # Find k-points grid using Monkorst-Pack method based on K-space resolution
                    #Get real space cell
                    cell = atoms.cell
                    
                    #Find Kpoints grid
                    #TODO: use structure_type info to generalize to non-periodic systems (3d, 2d, 1d, 0d)
                    MP_mesh = self._compute_kpoints_grid(cell, Kspace_resolution)

                    #Format k-points lines
                    kpoints_lines.append(f"\nK_POINTS automatic\n") #Header for MP-grid
                    Kpoint_line = f"{MP_mesh[0]} {MP_mesh[1]} {MP_mesh[2]}" #K-points grid line
                    for offset in Koffset: #Add offset
                        if offset: Kpoint_line += " 1"
                        else: Kpoint_line += " 0"
                    Kpoint_line += "\n"
                    kpoints_lines.append(Kpoint_line)

            # K_POINTS is defined by user in reference pwi file, keep the line/s
            elif idx_kpoints_line > 0:
                if 'gamma' in tmp_pwi_lines[idx_kpoints_line] or 'Gamma' in tmp_pwi_lines[idx_kpoints_line]: # KPOINT is 1 line
                    kpoints_lines = tmp_pwi_lines[idx_kpoints_line:idx_kpoints_line+1]
                    del tmp_pwi_lines[idx_kpoints_line:]
                elif 'automatic' in tmp_pwi_lines[idx_kpoints_line]: # KPOINTS is 2 lines
                    kpoints_lines = tmp_pwi_lines[idx_kpoints_line:idx_kpoints_line+2]
                    del tmp_pwi_lines[idx_kpoints_line:]
                elif 'tpiba' in tmp_pwi_lines[idx_kpoints_line] or 'crystal' in tmp_pwi_lines[idx_kpoints_line]: #KPOINTS is multiple lines
                    num_ks = int(tmp_pwi_lines[idx_kpoints_line+1].split()[0]) #Get number of k-points
                    kpoints_lines = tmp_pwi_lines[idx_kpoints_line:idx_kpoints_line+num_ks+2] #Get k-points lines
                    del tmp_pwi_lines[idx_kpoints_line:]
                else:
                    raise ValueError(f"K_POINTS format: {tmp_pwi_lines[idx_kpoints_line]} is unknown in pwi template file")
            
            return kpoints_lines
    
    def _compute_kpoints_grid(self, cell, Kspace_resolution):
        """
        Compute the k-points grid using Monkhorst-Pack method based on the cell and K-space resolution.
        """
        #Compute the reciprocal cell vectors: b_i = 2π * (a_j x a_k) / (a_i . (a_j x a_k))
        rec_cell = 2.0 * np.pi * np.linalg.inv(cell).T  

        #Compute reciprocal lattice vecotors' lenghts
        lengths = np.linalg.norm(rec_cell, axis=1)  

        #Compute mesh size
        mesh = [int(np.ceil(L / Kspace_resolution)) for L in lengths]  
        print(f"Computed k-points mesh: {mesh} for K-space resolution: {Kspace_resolution} Angstrom^-1") #DEBUG

        return mesh

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
        worker_output = {'success' : [], 'output' : [], 'outdir' : []}
        for pwi in pwi_files:
            #Try locking the pwi file
            lock_pwi, pwo_fname = self.lock_input(pwi_fname=pwi, worker_id=id)

            if lock_pwi == "": continue #Skip to next pwi if lock failed

            #Get output directory of this calculation
            with open(lock_pwi, 'r') as f:
                pwi_lines = f.readlines()
            outdir_line = [line.split('=')[1] for line in pwi_lines if 'outdir' in line][0]
            outdir_line = outdir_line.strip().replace("'", "").replace('"', '')  # Remove quotes
            outdir = os.getcwd() + f"/{outdir_line}"

            #Launch QE calculation
            success = self.run_qe(command=command, fname_pwi=lock_pwi, fname_pwo=pwo_fname)

            #Update output
            worker_output['success'].append(success)
            worker_output['output'].append(pwo_fname)
            worker_output['outdir'].append(outdir)

        return worker_output

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
        if os.path.exists(pwo_fname): return pwi_lock_fname, pwo_fname #If exists, skip to next pwi

        # Try to lock the pwi file by renaming it
        pwi_lock_fname = f'{pwi_fname}.lock_{worker_id}'
        try:
            os.rename(f'{pwi_fname}', f'{pwi_lock_fname}')
        except Exception as e:
            pwi_lock_fname = ""  
        
        return pwi_lock_fname, pwo_fname