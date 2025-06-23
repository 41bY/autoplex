import os
import subprocess
from dataclasses import dataclass
from ase.io import read, write
from jobflow import job, Flow, Maker, Response

def params_from_config(config: dict):
    """
    Return MatterGenMaker params from a configuration dictionary.

    Args:
        config (dict): Keys should match __init__ parameters. For example:
            {
                "results_path": "/path/to/results",
                "model_path": "/path/to/model",
                "batch_size": 128,
                "num_batches": 1,
                "record_trajectories": False
            }

    Returns:
        MatterGenMaker: Initialized instance.
    """
    #Get default parameters
    params = {
        "name": "MatterGenMaker",
        "results_path": "generated_structures",
        "model_path": None,
        "batch_size": 32,
        "num_batches": 1,
        "record_trajectories": False,
        "properties_to_condition_on": None,
        "diffusion_guidance_factor": None
    }    

    # Update parameters with values from the config file
    if config is None: raise ValueError("Configuration file is empty or not properly formatted.")
    params.update(config)

    #Check a valid model path is provided
    if not os.path.exists(params["model_path"]): raise ValueError(f"Model path '{params['model_path']}' not found.")

    return params

@dataclass
class MatterGenMaker(Maker):
    """
    Maker to set up MatterGen and run its generation process.
    Parameters:
        name (str): Name of the Maker.
        results_path (str): Path where results will be stored.
        model_path (str): Path to the generative model.
        batch_size (int): Number of samples per batch.
        num_batches (int): Total number of batches to generate.
        record_trajectories (bool): Whether to record trajectories.
        properties_to_condition_on (str): Properties to condition the generation on.
        diffusion_guidance_factor (float): Factor for diffusion guidance.
    """

    name: str = "MatterGenMaker"
    results_path: str | None = None
    model_path: str | None = None
    batch_size: int = 32
    num_batches: int = 1
    record_trajectories: bool = False
    properties_to_condition_on: str = None
    diffusion_guidance_factor: float = None

    def make(self, run_cmd: str | None = None):
        """
        Create a Flow to run the MatterGen generation command.
        
        Returns:
            Flow: A JobFlow that executes the MatterGen generation command.
        """
        #Define joblist
        joblist = []

        #Map config parameters to MatterGen's CLI arguments
        cli_cmds = self.config2cli(run_cmd)

        #Run the MatterGen-generate (possibly multiple instances TODO)
        extxyz_structures_path = self.run_mattergen(cli_cmds)
        extxyz_structures_path.name = "MatterGenGenerate"
        joblist.append(extxyz_structures_path)

        #Run the enumeration of generated structures
        generated_structures_paths = [extxyz_structures_path.output]
        enumerated_structures_path = self.enumerate_structures(generated_structures_paths)
        enumerated_structures_path.name = "EnumerateStructures"
        joblist.append(enumerated_structures_path)

        #Create the Flow to encapsulate the jobs
        matgen_flow = Flow(jobs=joblist, output=enumerated_structures_path.output)

        return Response(replace=matgen_flow, output=matgen_flow.output)
    
    def config2cli(self, run_cmd: str | None = None):
        """
        Convert MatterGen configuration parameters to command line arguments.
        
        Returns:
            list: List of command line arguments for MatterGen.
        """
        cmd = [
            'mattergen-generate',
            f'{self.results_path}',
            f'--model_path={self.model_path}',
            f'--batch_size={self.batch_size}',
            f'--num_batches={self.num_batches}',
            f'--record_trajectories={str(self.record_trajectories)}'
        ]

        if self.properties_to_condition_on is not None:
            cmd.append(f'--properties_to_condition_on="{self.properties_to_condition_on}"')
        if self.diffusion_guidance_factor is not None:
            cmd.append(f'--diffusion_guidance_factor={self.diffusion_guidance_factor}')
        
        if run_cmd is not None: cmd = [run_cmd] + cmd

        return cmd

    @job
    def run_mattergen(self, cmd: list[str]):
        """
        Build and execute the mattergen-generate command.
        """
        #Get output file path containing the generated structures
        cwd = os.getcwd()
        extxyz_structures_path = os.path.join(cwd, self.results_path, "generated_crystals.extxyz")

        #Run the commands
        cmd_line = ' '.join(cmd)
        try:
            subprocess.run([cmd_line], shell=True, check=True, executable="/bin/bash")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to run MatterGen command: {e}")

        return extxyz_structures_path

    @job
    def enumerate_structures(self, structures: list[str]):
        """
        Enumerate the generated structures.
        
        Args:
            structures (list[str]): List of paths to the generated structures.
        
        Returns:
            list[str]: List of enumerated structure paths.
        """
        # Read generated structures from the extxyz file
        atoms = []
        for structure in structures:
            atoms += read(structure, format="extxyz", index=":")
            
        # Label the structures with their unique indices
        for i, atom in enumerate(atoms):
            atom.info['unique_index'] = i
        
        # Dump enumerated structures back to extxyz format
        enumerated_structures_path = os.getcwd() + "/enumerated_structures.extxyz"
        write(enumerated_structures_path, atoms, format="extxyz", write_info=True)

        # Return the path to the enumerated structures
        return [enumerated_structures_path]