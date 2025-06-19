"""Jobs to create training data for ML potentials."""
import os
import logging
from glob import glob
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from ase import Atoms
from ase.io import read, write
from jobflow import Maker

def data_ensembler_from_config(config: dict):
    """
    Return DatasetMaker params from a configuration dictionary.

    Args:
        config (dict): Keys should match __init__ parameters. For example:
        {
            "labeled_output": labeled_output,
            "num_models": num_models,
            "test_ratio": test_ratio,
            "distill_force_max": distill_force_max,
            "force_label": force_label,
            "energy_label": energy_label,
            "output_file_name": output_file_name,
            "scf_code": scf_code,
            "pre_database_dir": pre_database_dir,
            "init_database_dir": init_database_dir,
            "isolated_atom_energies": isolated_atom_energies
        }

    Returns:
        params: dict
            Dictionary with parameters for QEstaticLabelling.
    """
    #Get default parameters
    #Collect parameters for DatasetMaker
    params = {
        "labeled_output": None,
        "scf_code": "QE",  # SCF code used for calculations, e.g., "QE" or "VASP"
        "num_models": 1,
        "test_ratio": 0.05,
        "distill_force_max": 30.0,
        "force_label": "REF_forces",
        "energy_label": "REF_energy",
        "output_file_name": "unique_dataset.extxyz",
        "pre_database_dir": None,
        "init_database_dir": None,
        "isolated_atom_energies": None
    } 

    # Update parameters with values from the config file
    if config is None: raise ValueError("Configuration file is empty or not properly formatted.")
    params.update(config)

    #Check labeled data path is provided
    if params["labeled_output"] is None: raise ValueError(f"Please provide a path or a list of path containing SCF labeled data files.")

    return params

@dataclass
class DatasetMaker(Maker):
    """
    Maker to prepare dataset from ab-initio calculations.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    num_models : int
        Number of ML-models in the ensemble. How many cross-validation folds to use.
    labeled_data_file: str
        Path to the file containing the DFT calculations data. It must be an ASE-readable file.
    test_ratio: float
        The proportion of the test set in the cross-validation splitting of the data.
    distill_force_max: float
        Maximum force value to exclude structures.
    force_label: str
        The label of force values saved inside the dataset.
    energy_label: str
        The label of energy values saved inside the dataset.
    pre_database_dir: str
        Directory where the previous database was saved.
    isolated_atom_energies: dict
        A dictionary containing isolated energy values for different species.
    """
    name: str = "do_dataset_preparation"
    labeled_output: str | list | None = None
    scf_code: str = "QE"  # SCF code used for calculations, e.g., "QE" or "VASP"
    num_models: int = 1
    test_ratio: float = 0.1
    distill_force_max: float | None = None
    force_label: str = "REF_forces"
    energy_label: str = "REF_energy"
    output_file_name: str | None = "unique_dataset.extxyz"
    pre_database_dir: str | None = None
    init_database_dir: str | None = None
    isolated_atom_energies: dict | None = None

    def make(self) -> Path:
        """
        Create the dataset for training ML potentials.

        Returns
        -------
        Path
            The current working directory.
        """
        #Collect labeled data from current iteration
        raw_atoms = self.load_labeled_data(
            labeled_outputs=self.labeled_output, 
            scf_code=self.scf_code
        )
        
        #Check if the labeled data is empty
        if not raw_atoms:
            msg = """No labeled data found.\n
                If this is the expected behavior, your MLIP model may have explored sufficiently the chemical space.\n
                Otherwise, please check the 'labeled_output' parameter.\n"""
            raise ValueError(msg)

        #Distill labeled data, excluding structures with forces larger than force_max
        if self.distill_force_max is not None:
            atoms = self.data_distillation(
                raw_atoms, 
                self.distill_force_max, 
                self.force_label)
        else:
            atoms = raw_atoms
        
        #Load initial dataset if provided
        if self.init_database_dir is not None and os.path.exists(self.init_database_dir):
            #Read initial dataset
            init_data = read(self.init_database_dir, index=":")
        else: init_data = []
        
        #Load previous iteration dataset if provided
        if self.pre_database_dir is not None and os.path.exists(self.pre_database_dir):
            #Read previous dataset
            pre_data = read(self.pre_database_dir, index=":")
        else: pre_data = []

        #Previous iteration dataset is the union of init_data (if provided) and data generated during the iterations
        #We need to remove initial dataset from previous iteration dataset before stratify and split the dataset
        if pre_data:
            if init_data:
                #Remove initial dataset from previous iteration dataset
                pre_data = [at for at in pre_data if at not in init_data]
            else:
                #If no initial dataset is provided, previous iteration dataset is entirely generated data during iterations
                pre_data = pre_data
        
        #Update data generated during previous + current iterations
        atoms = pre_data + atoms
        
        #Perform stratified dataset split with cross-validation
        ase_dataset, train_index_folds, test_index_folds = self.stratified_dataset_split_ensemble(
                atoms=atoms, 
                num_models=self.num_models,
                split_ratio=self.test_ratio, 
                energy_label=self.energy_label
                )        
        
        #Write the splitted dataset to files
        unique_dataset_path = self.write_splitted_dataset(
            output_file_name=self.output_file_name,
            ase_dataset=ase_dataset, 
            train_index_folds=train_index_folds, 
            test_index_folds=test_index_folds,
            init_data=init_data,
        )
        
        #Return the list of directories where the splitted indeces of the dataset are saved
        return unique_dataset_path

    def load_labeled_data(self, labeled_outputs: str | list[str], scf_code="QE") -> list[Atoms]:
        """
        Load labeled data from the provided file or list of files.

        Parameters
        ----------
        labeled_output: str | list[str]
            Path to the file containing the DFT calculations data or a list of such paths.
        scf_code: str
            The code used for the SCF calculations (e.g., "QE", "VASP", etc.).
        Returns
        -------
        raw_atoms: list[Atoms]
            List of ASE Atoms objects loaded from the labeled output.
        """
        if isinstance(labeled_outputs, str):
            #Check file exists
            if not os.path.exists(labeled_outputs):
                raise FileNotFoundError(f"The labeled output file {labeled_outputs} does not exist.")
            raw_atoms = read(labeled_outputs, index=":")

        elif isinstance(labeled_outputs, list):
            # Safe-search for labeled output file
            # Get every file in the same parent folder as the labeled_outputs
            output_files = []
            for labeled_output in labeled_outputs:
                try:
                    output_files += [pwo for success, pwo in zip(labeled_output['success'], labeled_output['pwo']) if success]
                except:
                    logging.error(f"Error in reading of labeled output: {labeled_output}, skipping it.")
            
            #Build unique output folders
            output_folders = set([os.path.dirname(file) for file in output_files])

            #Load code-dependent labeled outputs
            raw_atoms = self._load_code_specific_scf_output(output_folders, scf_code)
        
        return raw_atoms
    
    def _load_code_specific_scf_output(self, output_folders: set[str], scf_code: str) -> list[Atoms]:
        """
        Load SCF output files from the provided folders based on the SCF code.

        Parameters
        ----------
        output_folders: set[str]
            Set of folders containing the SCF output files.
        scf_code: str
            The code used for the SCF calculations (e.g., "QE", "VASP", etc.).

        Returns
        -------
        raw_atoms: list[Atoms]
            List of ASE Atoms objects loaded from the SCF output files.
        """
        #Get correct extension for the SCF code
        if scf_code == "QE":
            scf_extension = "*.pwo"
            ase_format = "espresso-out"
        elif scf_code == "VASP":
            scf_extension = "OUTCAR"
            ase_format = "vasp-out"
        else:
            raise ValueError(f"Unsupported SCF code: {scf_code}. Supported are QE | VASP. Please implement loading for this code.")

        #Get all SCF output files in the provided folders
        output_files = []
        for folder in output_folders:
            # Use glob to find all files matching the SCF extension
            output_files += glob(f"{folder}/{scf_extension}")

        #Read all SCF output files and return the raw atoms with labels
        raw_atoms = []
        for output_file in output_files:
            try:
                raw_atoms += read(output_file, index=":", format=ase_format)
            except Exception as e:
                logging.error(f"Skipping {output_file} due to error: {e}")
        
        return raw_atoms

    # TODO: Check 'regularization': what is it and how to use it?
    def write_splitted_dataset(self,
        output_file_name: str,                               
        ase_dataset: list[Atoms],
        train_index_folds: list[list[int]],
        test_index_folds: list[list[int]],                                 
        init_data: Atoms | list[Atoms] | None = None,
    ) -> Path:
        """
        Write the splitted dataset to files.
        This function is used to create a training and test set for each model in the ensemble.

        Parameters
        ----------
        output_file_name: str
            Name of the file where the unique ordered dataset will be saved.
        ase_dataset: list[Atoms]
            List of ASE Atoms objects, i.e. the dataset.
        train_index_folds: list[list[int]]
            List of training indices for each fold.
        test_index_folds: list[list[int]]
            List of test indices for each fold.
        init_data: Atoms | list[Atoms] | None
            Initial dataset to be appended to the training dataset. If None, no initial data is appended.
        
        Returns
        -------
        ensemble_dataset_dirs: list[str]
            List of directories where the splitted indeces of the dataset are saved.
        """

        #Append the whole initial dataset to the current training dataset
        if init_data:
            #Get previous dataset for training
            num_init_data = len(init_data)
            init_train_index = np.arange(len(ase_dataset), len(ase_dataset) + num_init_data) #Append indices for initial dataset 
            ase_dataset += init_data

            #Update indeces and structures datasets for each fold
            train_index_folds = [
                np.concatenate((train_index, init_train_index))
                for train_index in train_index_folds
            ]
        
        #Write unique ordered ase dataset
        dataset_dir = os.path.join(os.getcwd(), "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        unique_dataset_fname = os.path.join(dataset_dir, output_file_name)
        write(unique_dataset_fname, ase_dataset, format="extxyz", parallel=False)


        #Write cross-validation splitting indeces for each fold (i.e. model in the ensemble)
        header_line = f"indices refered to unique datset {unique_dataset_fname}"
        for fold_id, (train_index_fold, test_index_fold) in enumerate(zip(train_index_folds, test_index_folds)):
            #Get model directory
            model_dir = os.path.join(dataset_dir, f"NN{fold_id}")
            os.makedirs(model_dir, exist_ok=True)

            #Write cross-validation indeces to files
            train_index_fname, test_index_fname = os.path.join(model_dir, f"train_index.txt"), os.path.join(model_dir, f"test_index.txt")
            np.savetxt(train_index_fname, train_index_fold, fmt="%d", header=f"Train {header_line}")
            np.savetxt(test_index_fname, test_index_fold, fmt="%d", header=f"Test {header_line}")

        return unique_dataset_fname

    def stratified_dataset_split_ensemble(self,
        atoms: list[Atoms],
        num_models: int, 
        split_ratio: float, 
        energy_label: str,
    ) -> tuple[list[Atoms], list[int], list[int]]:
        """
        Apply stratified splitting to the dataset using (num_models)-fold cross-validation.
        This function is used to create a training and test set for each model in the ensemble.

        Parameters
        ----------
        atoms: Atoms
            ASE Atoms object
        num_models: int
            Number of models to be used in the ensemble. Obtain a (num_models)-fold cross-validation split.
        split_ratio: float
            Parameter to divide the training set and the test set.
        energy_label: str
            The label for the energy property in the atoms.

        Returns
        -------
        unique_ordered_ase_dataset: list[Atoms]
            List of unique ordered ASE Atoms objects.
        train_index_folds: list[list[int]]
            List of training indices for each fold.
        test_index_folds: list[list[int]]
            List of test indices for each fold.
        """
        atom_bulk = []
        atom_isolated_and_dimer = []
        for at in atoms:
            if ( 'structure_type' not in at.info or 
                (at.info["structure_type"] != "dimer"
                and at.info["structure_type"] != "IsolatedAtom")
            ):
                atom_bulk.append(at)
            else:
                atom_isolated_and_dimer.append(at)

        if len(atoms) != len(atom_bulk):
            atoms = atom_bulk

        # Need this try except block because the energy label is not present as info
        try:
            average_energies = np.array(
                [atom.info[energy_label] / len(atom) for atom in atoms]
            )
        except KeyError:
            average_energies = np.array(
                [atom.get_potential_energy() / len(atom) for atom in atoms]
            )
        # Sort atoms by increasing average energy
        sorted_indices = np.argsort(average_energies)
        atoms = [atoms[i] for i in sorted_indices]
        average_energies = average_energies[sorted_indices]

        #Perform train-test ensemble splitting: stratified splitting on average energy or random splitting if not enough data
        train_index_folds, test_index_folds = self._energy_stratified_split(
                                                            atoms,
                                                            average_energies,
                                                            usr_num_quantiles=2,
                                                            split_ratio=split_ratio,
                                                            num_models=num_models
                                                            )
        
        #Define unique ordered atoms dataset
        ase_dataset = atoms

        # Append whole sets of isolated atoms and dimers to training set
        if atom_isolated_and_dimer:
            iso_and_dimer_index = np.arange(len(atoms), len(atoms) + len(atom_isolated_and_dimer))
            train_index_folds = [np.concatenate((train_index, iso_and_dimer_index)) for train_index in train_index_folds]
            ase_dataset = atoms + atom_isolated_and_dimer 

        return ase_dataset, train_index_folds, test_index_folds
    
    def _energy_stratified_split(self,
        atoms,
        average_energies,
        usr_num_quantiles=2,
        split_ratio=0.2,
        num_models=5
    ):
        
        # Limit number of quantiles if few available data
        num_frames = len(atoms)
        max_q = max(1, num_frames // 2)
        q = min(usr_num_quantiles, max_q)

        # Create quantiles based on average energies
        if q > 1:
            labels = pd.qcut(
                average_energies,
                q=q,
                labels=False,
                duplicates='drop'
            )
            
            # If there are bins with less than 2 elements, merge them
            uniq, cnts = np.unique(labels, return_counts=True)
            small_bins = uniq[cnts < 2]
            for sb in small_bins:
                if sb == uniq.min():
                    tgt = sb + 1
                elif sb == uniq.max():
                    tgt = sb - 1
                else:
                    low_cnt = cnts[uniq == sb - 1][0]
                    high_cnt = cnts[uniq == sb + 1][0]
                    tgt = sb - 1 if low_cnt > high_cnt else sb + 1
                labels[labels == sb] = tgt
        else:
            labels = np.zeros(num_frames, dtype=int) # All atoms in one bin, if too few data

        # Compute number of test samples
        if isinstance(split_ratio, float):
            n_test = int(split_ratio * num_frames)
        else:
            n_test = int(split_ratio)
        
        # Ensure n_test is at least the number of unique labels
        if n_test < len(np.unique(labels)):
            n_test = len(np.unique(labels))
            logging.warning(f"Forcing number of test samples equal to the number of unique labels: {n_test}.")

        # Choice the splitter based on the number of unique labels
        if len(np.unique(labels)) < 2: # If only one unique label, use ShuffleSplit
            splitter = ShuffleSplit(
                n_splits=num_models,
                test_size=n_test,
                random_state=42
            )
            split_iter = splitter.split(atoms)
        else: # If multiple unique labels, use StratifiedShuffleSplit
            splitter = StratifiedShuffleSplit(
                n_splits=num_models,
                test_size=n_test,
                random_state=42
            )
            split_iter = splitter.split(atoms, labels)

        # Calculate train and test folds
        train_folds, test_folds = [], []
        for tr, ts in split_iter:
            train_folds.append(tr)
            test_folds.append(ts)

        return train_folds, test_folds

    def data_distillation(
            self,
            atoms: list[Atoms], 
            force_max: float, 
            force_label: str,
    ) -> list[Atoms]:
        """
        For data distillation.

        Parameters
        ----------
        labeled_data_file: str
            Path to the file containing the DFT calculations data. It must be an ASE-readable file.
        force_max: float
            Maximally allowed force.
        force_label: str
            The label for the force property in the atoms.

        Returns
        -------
        atoms_distilled:
            List of distilled atoms.

        """
        atoms_distilled = []
        for at in atoms:
            try:
                forces = np.abs(at.arrays[force_label])
            except KeyError:
                # If the force label is not found, use the default label
                forces = np.abs(at.get_forces())
            f_component_max = np.max(forces)

            if f_component_max < force_max:
                atoms_distilled.append(at)

        logging.warning(
            f"After distillation, there are still {len(atoms_distilled)} data points remaining."
        )

        return atoms_distilled