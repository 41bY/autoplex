"""Jobs to create training data for ML potentials."""
import os
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ase import Atom, Atoms
from ase.io import read, write
from jobflow import Maker


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
    labeled_data_file: str = "labels.extxyz"
    num_models: int = 1
    test_ratio: float = 0.1
    distill_force_max: float | None = None
    force_label: str = "REF_forces"
    energy_label: str = "REF_energy"
    output_file_name: str | None = "unique_dataset.extxyz"
    pre_database_dir: str | None = None
    isolated_atom_energies: dict | None = None

    def make(self) -> Path:
        """
        Create the dataset for training ML potentials.

        Returns
        -------
        Path
            The current working directory.
        """
        #TODO: Collect labeled data: from DFT-outputs to ASE-readable file excluding non-converged calculations (compositional energies)

        #Collect labeled data, excluding structures with forces larger than force_max
        if self.distill_force_max is not None:
            atoms = self.data_distillation(
                self.labeled_data_file, 
                self.distill_force_max, 
                self.force_label)
        else:
            atoms = read(self.labeled_data_file, index=":")
        
        #Perform stratified dataset split with cross-validation
        ase_dataset, train_index_folds, test_index_folds = self.stratified_dataset_split_ensemble(
                atoms=atoms, 
                num_models=self.num_models,
                split_ratio=self.test_ratio, 
                energy_label=self.energy_label
                )        
        
        #Write the splitted dataset to files
        ensemble_dataset_dirs = self.write_splitted_dataset(
            output_file_name=self.output_file_name,
            ase_dataset=ase_dataset, 
            train_index_folds=train_index_folds, 
            test_index_folds=test_index_folds,
            pre_database_dir=self.pre_database_dir
        )
        
        #Return the list of directories where the splitted indeces of the dataset are saved
        return ensemble_dataset_dirs

    # TODO: Check 'regularization': what is it and how to use it?
    def write_splitted_dataset(self,
        output_file_name: str,                               
        ase_dataset: list[Atoms],
        train_index_folds: list[list[int]],
        test_index_folds: list[list[int]],                                 
        pre_database_dir: str | None = None,
    ) -> list[str]:
        """
        Write the splitted dataset to files.
        This function is used to create a training and test set for each model in the ensemble.

        Parameters
        ----------
        ase_dataset: list[Atoms]
            List of ASE Atoms objects, i.e. the dataset.
        train_index_folds: list[list[int]]
            List of training indices for each fold.
        test_index_folds: list[list[int]]
            List of test indices for each fold.
        pre_database_dir: str | None
            Directory where the previous database was saved.
            If None, the previous database will not be used.
        
        Returns
        -------
        ensemble_dataset_dirs: list[str]
            List of directories where the splitted indeces of the dataset are saved.
        """

        #Append previous training and testing datasets
        #TODO: Handling pre_database_dir?
        if pre_database_dir and os.path.exists(pre_database_dir):
            #Get previous dataset for training
            pre_data_train = read(
                os.path.join(pre_database_dir, "train.extxyz"), index=":"
            )
            pre_train_index = np.arange(len(ase_dataset), len(ase_dataset) + len(pre_data_train))
            ase_dataset += pre_data_train

            #Get previous dataset for testing
            pre_data_test = read(
                os.path.join(pre_database_dir, "test.extxyz"), index=":"
            )
            pre_test_index = np.arange(len(ase_dataset), len(ase_dataset) + len(pre_data_test))
            ase_dataset += pre_data_test

            #Update indeces and structures datasets for each fold
            train_index_folds = [
                np.concatenate((train_index, pre_train_index))
                for train_index in train_index_folds
            ]
            test_index_folds = [
                np.concatenate((test_index, pre_test_index))
                for test_index in test_index_folds
            ]
        
        #Write unique ordered ase dataset
        dataset_dir = os.path.join(os.getcwd(), "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        unique_dataset_fname = os.path.join(dataset_dir, output_file_name)
        write(unique_dataset_fname, ase_dataset, format="extxyz", parallel=False)


        #Write cross-validation splitting indeces for each fold (i.e. model in the ensemble)
        ensemble_dataset_dirs = []
        header_line = f"indices refered to unique datset {unique_dataset_fname}"
        for fold_id, (train_index_fold, test_index_fold) in enumerate(zip(train_index_folds, test_index_folds)):
            #Get model directory
            model_dir = os.path.join(dataset_dir, f"NN{fold_id}")
            os.makedirs(model_dir, exist_ok=True)

            #Write cross-validation indeces to files
            train_index_fname, test_index_fname = os.path.join(model_dir, f"train_index.txt"), os.path.join(model_dir, f"test_index.txt")
            np.savetxt(train_index_fname, train_index_fold, fmt="%d", header=f"Train {header_line}")
            np.savetxt(test_index_fname, test_index_fold, fmt="%d", header=f"Test {header_line}")

            #Save paths to the model directories
            ensemble_dataset_dirs.append(model_dir)

        return ensemble_dataset_dirs

    def stratified_dataset_split_ensemble(self,
        atoms: Atoms,
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
            if (
                at.info["structure_type"] != "dimer"
                and at.info["structure_type"] != "IsolatedAtom"
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
        # sort by energy
        sorted_indices = np.argsort(average_energies)
        atoms = [atoms[i] for i in sorted_indices]
        average_energies = average_energies[sorted_indices]

        # Perform cross-validation stratified (on avg. energy) splitting
        num_quantiles = 2
        stratified_average_energies = pd.qcut(average_energies, q=num_quantiles, labels=False)

        # Check that test_size is >= number_of_stratified_classes
        if int(split_ratio * len(atoms)) < num_quantiles:
            split_ratio = int(num_quantiles)
            logging.warning(f"Test size ratio is too small. Setting test_size to number of quantiles = {split_ratio}.")
        
        # Create stratified train-test split
        split = StratifiedShuffleSplit(n_splits=num_models, test_size=split_ratio, random_state=42)
        # Return stratified train-test split indeces for each fold
        train_index_folds, test_index_folds = [], []
        for train_index, test_index in split.split(atoms, stratified_average_energies):
            #Update the folds
            train_index_folds.append(train_index), test_index_folds.append(test_index)
        
        #Define unique ordered atoms dataset
        ase_dataset = atoms

        # Append isolated atoms and dimers to training set
        if atom_isolated_and_dimer:
            iso_and_dimer_index = np.arange(len(atoms), len(atoms) + len(atom_isolated_and_dimer))
            train_index_folds = [np.concatenate((train_index, iso_and_dimer_index)) for train_index in train_index_folds]
            ase_dataset = atoms + atom_isolated_and_dimer 

        return ase_dataset, train_index_folds, test_index_folds

    def data_distillation(self,
        labeled_data_file: str, force_max: float, force_label: str
    ) -> list[Atom | Atoms]:
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
        atoms = read(labeled_data_file, index=":")

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