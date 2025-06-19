"""Flows consisting of jobs to fit ML potentials."""

import os
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.io import read, write
from jobflow import Flow, Maker, job, Response

from autoplex import MLIP_HYPERS
from autoplex.fitting.common.utils import mace_fitting, check_convergence

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="mlip_fitting.log"
)


#Maker to perform ensemble MLIP fitting
#TODO: hyperparams optimization using XOPT
@dataclass
class MLIPEnsembleMaker(Maker):
    """
    Maker to fit ensemble of ML potentials based on DFT labelled reference data.

    This Maker will filter the provided dataset in a data preprocessing step and then proceed
    with the MLIP fit (default is GAP).

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    dataset_path: str
        List containing path to dataset folder.        
    mlip_type: Literal["MACE"]
        Choose one specific MLIP type to be fitted.
    mlip_train_kwargs: dict
        Dictionary containing hyperparameters for the MLIP training.
    num_models: int
        Number of models in the ensemble to be fitted.
    remove_model_datasets: bool
        Whether to remove datasets specific to each model in the ensemble after training.
    """

    name: str = "Ensemble_MLIP_Fit"
    dataset_path: str | None = None
    mlip_type: Literal["MACE"] = "MACE"
    mlip_train_kwargs: dict = field(default_factory=dict)
    num_models: int = 1
    remove_model_datasets: bool = True


    def make(self):
        """
        Make a flow for fitting an ensemble of MLIP models.

        Parameters
        ----------
        database_dir: Path | str
            Path to the directory containing the datasets.
        hyperparameters: MLIP_HYPERS
            Hyperparameters for the MLIP.
        fit_kwargs: dict
            Additional keyword arguments for MLIP fitting.
        remove_model_datasets: bool
            Whether to remove datasets specific to each model in the ensemble after training.
        """

        #Check the right type of MLIP
        if self.mlip_type not in ["MACE"]:
            raise ValueError(
                "Please correct the MLIP name!"
                "The current version ONLY supports the following models: MACE."
            )

        #Define output of this workflow
        training_jobs = []
        output = {"mlip_paths": [], "train_errors": [], "test_errors": [], "convergences": []}

        #Load dataset
        ase_dataset = read(self.dataset_path, index=":")     

        #Get the top-level database directory
        root_database_dir = os.path.dirname(self.dataset_path)

        #Assert number of models equal number of train-test dirs
        model_dirs = [dir for dir in os.listdir(root_database_dir) if dir.startswith("NN")]
        assert len(model_dirs) == self.num_models, f"Number of models {self.num_models} does not match number of model directories {len(model_dirs)} in {root_database_dir}."
        
        #Setup training directories
        for model_index in range(self.num_models):
            #Get data sub-directory for each model
            model_dataset_dir = os.path.join(root_database_dir, f"NN{model_index}")

            #Write cross-validated train and test data to the model directory
            train_fname, test_fname = self.write_mlip_dataset(
                outdir=model_dataset_dir,
                mlip_type=self.mlip_type,
                dataset=ase_dataset,
            )
        
            #Write mace_input file
            train_input_fname = self.write_mlip_input(
                outdir=model_dataset_dir,
                mlip_type=self.mlip_type,
                usr_hypers=self.mlip_train_kwargs,
                train_fname=train_fname,
                test_fname=test_fname,
                )

            #Run MLIP fitting (job)
            mlip_train_job = self.mlip_training(
                input_fname=train_input_fname,
                type=self.mlip_type,
                train_data_path=train_fname,
                test_data_path=test_fname,
                )

            # Append training job to the list of jobs and outputs job to the output dictionary
            mlip_train_job.name = f"training_mlip_{model_index}"
            training_jobs.append(mlip_train_job)
            output["mlip_paths"].append(mlip_train_job.output["mlip_path"])
            output["train_errors"].append(mlip_train_job.output["train_error"])
            output["test_errors"].append(mlip_train_job.output["test_error"])
            output["convergences"].append(mlip_train_job.output["convergence"])

        #Define ensemble training flow
        training_flow = Flow(
            jobs=training_jobs,
            name="training_mlip_flow",
            output=output,
        )

        return Response(
            replace=training_flow,
            output=training_flow.output,
        )


    def write_mlip_dataset(self, outdir: str, mlip_type: str, dataset: list[Atoms]) -> tuple[str, str]:
        """
        Write the specific dataset for MLIP model in the ensemble.

        Parameters
        ----------
        dataset: list[Atoms]
            List of ASE Atoms objects containing the training data.
        """
        #Get cross-validated train and test data
        train_idx_file = os.path.join(outdir, "train_index.txt")
        test_idx_file = os.path.join(outdir, "test_index.txt")
        train_index, test_index = np.loadtxt(train_idx_file, dtype=int, ndmin=1), np.loadtxt(test_idx_file, dtype=int, ndmin=1)

        #Train-test splitting
        train_data = [dataset[i] for i in train_index]
        test_data = [dataset[i] for i in test_index]        

        #Write train and test dataset specific for model type
        if mlip_type != "MACE":
            raise ValueError(
                "The current version ONLY supports the following models: MACE."
            )

        #Write train and test data to files
        train_fname = os.path.join(outdir, "train.extxyz")
        write(train_fname, train_data, format="extxyz")

        test_fname = os.path.join(outdir, "test.extxyz")
        write(test_fname, test_data, format="extxyz")
        
        logging.info(f"Written train file to: {train_fname}")
        logging.info(f"Written test file to: {test_fname}")

        return train_fname, test_fname


    def write_mlip_input(self, outdir: str, mlip_type: str, usr_hypers: dict, train_fname: str, test_fname: str) -> str:
        """
        Write the MACE input file.

        Parameters
        ----------
        hypers: dict
            Dictionary containing hyperparameters required for the MACE model training.
        """
        if mlip_type == "MACE":
            # Define default hyperparameters
            hypers = {
                "name": "MACE",
                "model": "MACE",
                "r_max": 6.0,
                "correlation": 3,
                "num_channels": 16,
                "num_interactions": 2,
                "max_L": 1,
                "max_ell": 3,
                "distance_transform": "Agnesi",
                "lr": 0.01,
                "amsgrad": True,
                "ema": True,
                "ema_decay": 0.99,
                "patience": 30,
                "scheduler_patience": 2,
                "max_num_epochs": 500,
                "start_swa": 450,
                "swa": True,
                "seed": 12345,
                "train_file": "",
                "valid_file": "",
                "E0s": None,
                "energy_key": "REF_energy",
                "energy_weight": 1.0,
                "forces_key": "REF_forces",
                "forces_weight": 100.0,
                "num_workers": 8,
                "batch_size": 16,
                "default_dtype": "float32",
                "device": "cuda",
                "distributed": False,
                "enable_cueq": True,
                "restart_latest": True,
            }
        
        else:
            raise ValueError(
                "Please correct the MLIP name!"
                "The current version ONLY supports the following models: MACE."
            )

        # # Substitute default hyperparameters with user-defined ones
        hypers.update(usr_hypers)

        #Set name of the model
        hypers["name"] = "MACE"               
        
        #Set training and test data files
        hypers["train_file"], hypers["valid_file"] = train_fname, test_fname

        #Set output file name
        out_fname = os.path.join(outdir, "input.yaml")

        #Write hyperparameters to file
        with open(out_fname, "w") as f:
            yaml.safe_dump(hypers, f, sort_keys=False)
        logging.info(f"Written {mlip_type} input file to: {out_fname}")

        #Return path to written input file
        return out_fname
    
    @job
    def mlip_training(self,
        input_fname: str,
        type: str,
        train_data_path: str,
        test_data_path: str,
    ):
        """
        Job for fitting potential(s).

        Parameters
        ----------
        database_dir: Str | Path
            Path to the directory containing the database.
        species_list: list
            List of element names (strings) involved in the training dataset
        run_fits_on_different_cluster: bool
            Whether to run fitting on different clusters.
        isolated_atom_energies: dict
            Dictionary of isolated atoms energies.
        num_processes_fit: int
            Number of processes for fitting.
        auto_delta: bool
            Automatically determine delta for 2b, 3b and soap terms. Only used for GAP fitting.
        glue_xml: bool
            Use the glue.xml core potential instead of fitting 2b terms. Only used for GAP fitting.
        glue_file_path: str
            Name of the glue.xml file path. Only used for GAP fitting.
        gpu_identifier_indices: list[int]
            List of GPU indices to be used for fitting. Only used for NEP fitting.
        mlip_type: str
            Choose one specific MLIP type to be fitted:
            'GAP' | 'J-ACE' | 'NEQUIP' | 'NEP' | 'M3GNET' | 'MACE'
        ref_energy_name: str
            Reference energy name.
        ref_force_name: str
            Reference force name.
        ref_virial_name: str
            Reference virial name.
        device: str
            Device to be used for model fitting, either "cpu" or "cuda".
        database_dict: dict
            Dict including all training and test databases.
        hyperpara_opt: bool
            Perform hyperparameter optimization using XPOT
            (XPOT: https://pubs.aip.org/aip/jcp/article/159/2/024803/2901815)
        hyperparameters: MLIP_HYPERS
            Hyperparameters for MLIP fitting.
        run_fits_on_different_cluster: bool
            Indicates if fits are to be run on a different cluster.
            If True, the fitting data (train.extxyz, test.extxyz) is stored in the database.
        fit_kwargs: dict
            Additional keyword arguments for MLIP fitting.
        """
        if isinstance(input_fname, str):  # data_prep_job.output is returned as string
            input_fname = Path(input_fname)

        if type == "MACE":
            mlip_train_output = self.run_mace(input_fname)
        else:
            raise ValueError(
                "Please correct the MLIP name!"
                "The current version ONLY supports the following models: MACE."
            )

        return mlip_train_output


    def run_mace(self, fname_input) -> dict:
        """
        MACE runner.

        Parameters
        ----------
        fname_input: str
            Path to the MACE training input file.

        """
        # Define output values
        mlip_path, train_error, test_error, convergence = None, None, None, None
        
        #Get cmd string
        cmd = "mace_run_train"
        arg = f"--config {fname_input}"
        command = f"{cmd} {arg}"

        # Launch MACE training and wait for it to finish
        with (
            open("mace_train_out.log", "w", encoding="utf-8") as file_std,
            open("mace_train_err.log", "w", encoding="utf-8") as file_err,
        ):
            subprocess.run(command, stdout=file_std, stderr=file_err, shell=True)
        
        # Check if the MACE training was successful
        with open("mace_train_out.log", "r", encoding="utf-8") as file_std:
            lines = file_std.readlines()
        
        convergence = False
        for line in lines:
            if "Done" in line:
                convergence = True
                logging.info("MACE training completed successfully.")
                break

        if not convergence:
            raise RuntimeError(
                "MACE training failed. Please check the log files for more details."
            )
        
        # Get training and test errors and path to the trained model
        if convergence:
            log_lines = []
            with open("mace_train_out.log", "r", encoding="utf-8") as file_std:
                log_lines = file_std.readlines()
            
            train_errors, test_errors = [], []
            for line in log_lines:
                if "| train_Default |" in line or "| train_default |" in line:
                    train_errors.append(float(line.split()[5]))
                elif "| valid_Default |" in line or "| valid_default |" in line:
                    test_errors.append(float(line.split()[5]))
            
            # Get the path to the trained MACE model
            mace_stage_two = os.path.join(os.getcwd(), "MACE_stagetwo.model")
            mace_stage_one = os.path.join(os.getcwd(), "MACE.model")

            if os.path.exists(mace_stage_two):
                mlip_path = mace_stage_two
                train_error, test_error = train_errors[-1], test_errors[-1]
            elif os.path.exists(mace_stage_one):
                mlip_path = mace_stage_one
                train_error, test_error = train_errors[0], test_errors[0]
            else:
                raise FileNotFoundError(
                    "MACE model file not found. Please check the training output."
                )

        return {
            "mlip_path": mlip_path,
            "train_error": train_error,
            "test_error": test_error,
            "convergence": convergence,
        }            