"""Flows consisting of jobs to perform exploration based on model-deviation and sampling."""

import os
import logging
from typing import Literal
from dataclasses import dataclass, field

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

        # For each relaxed structure, evaluate the deviation of the ensemble of models
        # return list of ase atoms with: array['force_deviation'] and info['energy_deviation']
        relaxed_structures = self.evaluate_ensemble_deviation(
            structures=relaxed_structures,
            mlip_models=mlip_models,
        )

        # Sample the structures based on the deviation of the ensemble of models
        sampled_structures = self.sample_structures(
            structures=relaxed_structures,
            force_deviation_threshold=0.1, #TODO: Implement more sophisticated sampling methods
        )

        # Save the sampled structures to a file
        cwd = os.getcwd()
        sampled_structures_path = os.path.join(cwd, "sampled_structures.extxyz")
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
        else:
            pilot_model = mlip_model

        # Loop over structures
        relaxed_structures = []
        for structure in structures:
            # Set the calculator for the structure
            structure.calc = pilot_model
            
            # Relax the structure
            relaxation_dyn = BFGS(structure)
            relaxation_dyn.run(fmax=0.01, steps=1000) # Convergence criteria

            # Get the relaxed structure
            relaxed_structure = structure.copy()
            relaxed_structure.calc = None
            relaxed_structures.append(relaxed_structure)


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
            # Get the forces and energies from each model
            forces = []
            energies = []
            for mlip_model in mlip_models:
                structure.calc = mlip_model
                forces.append(structure.get_forces())
                energies.append(structure.get_potential_energy())

            # Calculate the deviation of the ensemble of models
            forces = np.array(forces)
            force_value = np.mean(forces, axis=0)
            force_deviation = np.std(forces, axis=0)

            energies = np.array(energies)
            energy_value = np.mean(energies, axis=0)
            energy_deviation = np.std(energies, axis=0)

            # Store the deviation in the structure
            structure.calc = None
            structure.forces = force_value
            structure.arrays["force_deviation"] = force_deviation
            structure.energy = energy_value
            structure.info["energy_deviation"] = energy_deviation

        return structures

    def sample_structures(
            self,
            structures: list[Atoms],
            force_deviation_threshold: float = 0.1,
            ):
        """
        Function to sample the structures based on the deviation of the ensemble of models.
        """
        # Loop over structures
        sampled_structures = []
        for structure in structures:
            # Get the force and energy deviation
            force_deviation = structure.arrays["force_deviation"]

            # Check if the deviation is above the threshold
            if np.max(force_deviation) > force_deviation_threshold:
                sampled_structures.append(structure)

        return sampled_structures

