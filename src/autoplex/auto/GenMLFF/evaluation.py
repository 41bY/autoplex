"""Flows consisting of jobs to perform exploration based on model-deviation and sampling."""

import os
import logging
from typing import Literal
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from ase.io import read, write
from jobflow import Maker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="mlip_fitting.log"
)


@dataclass
class EnsembleEvaluatorMaker(Maker):
    """
    Class to evaluate the ensemble of MLIP models.
    """
    name: str = "MLIP_ensemble_evaluator"
    mlip_type: str | None = None # Type of MLIP model to use: "MACE"
    mlip_paths: list[str] | None = None # Paths to the MLIP models
    mlip_errors: list[float] | None = None # Training errors of the MLIP models -> decide the pilot model
    mlip_kwargs: dict | None = None
    structure_paths: list[str] | None = None
    pre_trained_model: str | None = None
    pre_trained_kwargs: dict | None = None


    def make(self):
        """
        Function to create the MLIP ensemble evaluator.
        """
        # Load the ensemble of MLIP models: first model is the "pilot" model
        mlip_models = self.load_mlip_models(
            mlip_type=self.mlip_type,
            mlip_kwargs=self.mlip_kwargs,
            mlip_paths=self.mlip_paths,
            mlip_errors=self.mlip_errors,
        )

        # Load the structures to be evaluated
        structures = []
        for path in self.structure_paths:
            structures += read(path, index=":")

        # Relax the structures using the pilot model
        relaxed_structures = self.relax_structures(
            structures=structures,
            mlip_model=mlip_models[0],
            pre_trained_model_path=self.pre_trained_model,
            pre_trained_kwargs=self.pre_trained_kwargs,
        )

        #Load every relaxed structure to evaluate the ensemble deviation (it should work also in case of interrupted runs)
        fnames = [f"{os.getcwd()}/relax_id{structure.info['unique_index']}.extxyz" for structure in relaxed_structures]
        relaxed_structures = []
        for fname in fnames:
            try:
                atoms = read(fname)
                relaxed_structures.append(atoms)
            except Exception as e:
                logging.warning(f"Failed to read structure from {fname}: {e}")
                continue

        # For each relaxed structure, evaluate the deviation of the ensemble of models
        # return list of ase atoms with: array['force_deviation'] and info['energy_deviation']
        # This should be pretty fast, since the structures are already relaxed
        devi_structures = self.evaluate_ensemble_deviation(
            structures=relaxed_structures,
            mlip_models=mlip_models,
        )

        print(f"Evaluated the ensemble deviation for {len(devi_structures)} structures.") #DEBUG
        print(f"Force average shape = {[atoms.arrays['forces'].shape for atoms in devi_structures]}") #DEBUG
        print(f"Force average shape = {[atoms.arrays['force_avg'].shape for atoms in devi_structures]}") #DEBUG
        print(f"Force std shape = {[atoms.arrays['force_std'].shape for atoms in devi_structures]}") #DEBUG
        print(f"Force average shape = {[atoms.info['energy'] for atoms in devi_structures]}") #DEBUG
        print(f"Energy average shape = {[atoms.info['energy_avg'] for atoms in devi_structures]}") #DEBUG
        print(f"Energy std shape = {[atoms.info['energy_std'] for atoms in devi_structures]}") #DEBUG

        #Save the relaxed structures with the evaluated deviations
        write("evaluated_structures.extxyz", 
              devi_structures, 
              format="extxyz", 
              columns=['symbols', 
                       'positions',
                       'forces',
                       'force_avg',
                       'force_std'],
              write_info=True)

        #Remove saved structures with no deviations
        for fname in fnames:
            if not os.path.exists(fname): continue
            print(f"Removing original relaxed structure: {fname}") #DEBUG
            os.remove(fname) # Remove the original relaxed structures

        # Sample the structures based on the deviation of the ensemble of models
        #TODO: Implement more sophisticated sampling methods
        sampled_structures = self.select_structures(
            structures=devi_structures,
            deviation_threshold=0.1,
            relative=0.01, # Relative threshold for force deviation 
        )

        print(f"Sampled {len(sampled_structures)} structures using ensemble deviation.") #DEBUG

        # Save the sampled structures to a file
        cwd = os.getcwd()
        sampled_structures_path = os.path.join(cwd, "selected_structures.extxyz")
        write(sampled_structures_path, sampled_structures, format="extxyz")
        logging.info(f"Sampled structures saved to {sampled_structures_path}")

        return sampled_structures_path

    def load_mlip_models(
            self,
            mlip_type: str,
            mlip_kwargs: dict,
            mlip_paths: list[str],
            mlip_errors: list[float],
            ):
        """
        Function to load the ensemble of MLIP models.
        """
        # Load the MLIP models from the specified directory
        mlip_models = []
        for mlip_path in mlip_paths:
            if mlip_type == "MACE":
                # Load the MACE model
                mlip_model = MACECalculator(**mlip_kwargs, model_paths=mlip_path)
            else:
                raise ValueError(f"Unsupported MLIP type: {mlip_type}. Supported types are: MACE.")
            mlip_models.append(mlip_model)
        
        # Sort the models based on the error: lower errors first
        mlip_models = sorted(mlip_models, key=lambda x: mlip_errors[mlip_models.index(x)])

        return mlip_models
    
    def relax_structures(
            self,
            structures: list[Atoms],
            mlip_model: MACECalculator,
            pre_trained_model_path: str | None = None,
            pre_trained_kwargs: dict | None = None,
            ):
        """
        Function to relax a list of structures using the specified MLIP model.
        """
        # Get pilot model
        if pre_trained_model_path is not None:
            pilot_model = MACECalculator(model_paths=pre_trained_model_path, **pre_trained_kwargs)
            logging.info(f"Pilot model set to pre-trained model: {pre_trained_model_path}")
        else:
            pilot_model = mlip_model
            logging.info(f"Pilot model set to the best model in the ensemble")

        #Get working directory
        cwd = os.getcwd()

        # Loop over structures
        relaxed_structures = []
        for structure in structures:
            #Search for the structure in the working directory
            structure_fname = f"{cwd}/relax_id{structure.info['unique_index']}.extxyz"
            if os.path.exists(structure_fname): continue # Skip if the structure is already relaxed

            # Set the calculator for the structure
            structure.calc = pilot_model
            
            # Relax the structure
            relaxation_dyn = BFGS(structure)
            relaxation_dyn.run(fmax=0.01, steps=1000) # Convergence criteria

            # Append the relaxed structure
            relaxed_structures.append(structure)

            # Save the relaxed structure to a file
            write(structure_fname, structure, format="extxyz", write_info=True)

        return relaxed_structures
    
    def evaluate_ensemble_deviation(
            self,
            structures: list[Atoms],
            mlip_models: list[MACECalculator],
            ):
        """
        Function to evaluate the deviation of the ensemble of MLIP models.
        """
        # Loop over structures
        for structure in structures:
            # Get computed energy and forces from the pilot model
            energy = structure.get_potential_energy().copy()
            forces = structure.get_forces().copy()
            stress = structure.get_stress().copy()

            # Get the evaluated forces and energies from each model
            eval_energies, eval_forces = [], []
            for mlip_model in mlip_models:
                structure.calc = mlip_model
                eval_forces.append(structure.get_forces())
                eval_energies.append(structure.get_potential_energy())

            # Calculate average and std of forces over the ensemble of models
            eval_forces = np.array(eval_forces)
            force_avg = np.mean(eval_forces, axis=0)
            force_std = np.std(eval_forces, axis=0)

            # Calculate average and std of energy over the ensemble of models
            eval_energies = np.array(eval_energies)
            energy_avg = np.mean(eval_energies, axis=0)
            energy_std = np.std(eval_energies, axis=0)

            # Remove calculator from the structure to avoid confusion
            structure.calc = None

            # Insert back the results into the structure
            structure.info['energy'] = energy
            structure.arrays['forces'] = forces
            structure.info['stress'] = stress

            # Store averages and deviations in the structure
            structure.arrays["force_avg"] = force_avg
            structure.arrays["force_std"] = force_std
            structure.info["energy_avg"] = energy_avg
            structure.info["energy_std"] = energy_std

        return structures

    def select_structures(
            self,
            structures: list[Atoms],
            deviation_threshold: float = 0.1,
            relative: float | None = None,
            ):
        """
        Function to sample the structures based on the deviation of the ensemble of models.
        """
        # Loop over structures
        sampled_structures = []
        for structure in structures:
            # Get force values and deviations
            force_value = structure.arrays["force_avg"]
            force_deviation = structure.arrays["force_std"]

            if relative is not None:
                # Compute force relative deviation as metric
                # Using relative as a filter parameter for small forces
                deviation_metric = np.abs(force_value) / (np.abs(force_value) + relative)
            else:
                # Use absolute deviation as metric
                deviation_metric = np.abs(force_deviation)

            # Check if the maximum of deviation metric is above the threshold
            if np.max(deviation_metric) > deviation_threshold:
                sampled_structures.append(structure)

        return sampled_structures