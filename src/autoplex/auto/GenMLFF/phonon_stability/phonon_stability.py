import os
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import simpson

from ase.io import read, write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.phonon.dos import PhononDos

from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import ForceFieldStaticMaker, ForceFieldRelaxMaker

from maggma.stores import JSONStore, MemoryStore
from jobflow import Response, Flow, Maker, job, run_locally
from jobflow.core.store import JobStore 

#Set up logger
import logging

# 1) imposta globalmente il livello INFO (o DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="local_run.log",  # Log to a file
)

@dataclass
class StabilityPhononFlowMaker(Maker):
    """
    A Maker class to create a phonon stability flow for a set of structures.
    It uses a pre-trained force field for phonon calculations.
    Parameters:
        model_path (str): Path to the pre-trained MLIP model.
        model_name (str): Name of the MLIP model.
        model_kwargs (dict): Dictionary of model calculator kwargs.
        max_parallel_flows (int): Maximum number of parallel flows to run.
        compute_metrics (bool): Whether to compute stability metrics.
    """
    name = "StabilityPhononFlowMaker"
    model_path: str | None = None
    model_name: str | None = None
    model_kwargs: dict = field(default_factory=dict)
    max_parallel_flows: int | None = None
    compute_metrics: bool = False

    def make(
            self,
            phonon_kwargs: dict = field(default_factory=dict), 
            data_name: str | None = None, 
            data_path: str | None = None
        ):
        #Read the structures from the file
        structures = read(data_path, index=":")

        #Get phonon maker kwargs
        ph_maker_kwargs = phonon_kwargs

        #Define static maker kwargs
        st_maker_kwargs = {
            "force_field_name": f"MLFF.{self.model_name}", # MLFF.MACE, MLFF.MatterSim, MLFF.META_eqv2
            "calculator_kwargs": self.model_kwargs      
        }        
        #Define relax maker kwargs
        relax_maker_kwargs = {
            "force_field_name": f"MLFF.{self.model_name}",
            "calculator_kwargs": self.model_kwargs     
        }

        #Define joblist
        joblist, output = [], []

        #Assign a unique id to each structure
        structures_idxs = [idx for idx, _ in enumerate(structures)]

        #Divide the list of structures and idxs in batches if max_parallel_flows is set
        if self.max_parallel_flows is not None:
            #Get batch size
            batch_size = max(1, len(structures) // self.max_parallel_flows)
            num_batches = len(structures)//batch_size if len(structures) % batch_size == 0 else len(structures)//batch_size + 1
        else: 
            batch_size, num_batches = 1, len(structures)

        #Create batches of structures and idxs
        # (num_batches - 1) batches of size batch_size and one last batch with the remaining structures
        structure_batches = [
            structures[i * batch_size: (i + 1) * batch_size] for i in range(num_batches - 1)
        ] + [structures[(num_batches - 1) * batch_size:]]
        idxs_batches = [
            structures_idxs[i * batch_size: (i + 1) * batch_size] for i in range(num_batches - 1)
        ] + [structures_idxs[(num_batches - 1) * batch_size:]]        

        # Define one serial_phononflow job for each structure batch
        for structures, idxs in zip(structure_batches, idxs_batches):
            # Convert ASE Atoms objects to Pymatgen Structures
            pmg_structures = [AseAtomsAdaptor.get_structure(structure) for structure in structures]

            # Create a serial_phonon flow for that batch of structures
            serial_ph = self.local_phonon_flow(
                ph_makers_kwargs=ph_maker_kwargs, 
                st_makers_kwargs=st_maker_kwargs,
                relax_makers_kwargs=relax_maker_kwargs,
                pmg_structures=pmg_structures,
                idxs_structures=idxs,
                output_dir=f"results_{self.model_name}_{data_name}",  # Directory to store the results
            )
            if len(idxs) == 1:
                serial_ph.name = f"phonons_{self.model_name}_{data_name}_{idxs[0]}"
            else:
                serial_ph.name = f"phonons_{self.model_name}_{data_name}_{idxs[0]}-{idxs[-1]}"

            #Populate the joblist and output
            joblist.append(serial_ph), output.append(serial_ph.output) #output is a list (batches) of list (batched_structures) of dict (result)
        #Assemble the phonons flow
        phonons_flow = Flow(joblist, output, name=f"ph_{self.model_name}_{data_name}_stability")
        
        if self.compute_metrics:
            #Compute dynamic stability metric for each structure
            metric_job = self.compute_stability_metrics(phonons_flow.output, neg_modes_threshold=0.01)
            metric_job.name = "stability_metrics_job"

            #Assemble the stability flow
            final_flow = Flow([phonons_flow, metric_job], metric_job.output, name=f"ph_{self.model_name}_{data_name}_stability")
        else:
            final_flow = phonons_flow

        return Response(replace=final_flow)

    @job
    def local_phonon_flow(
        self,
        ph_makers_kwargs : dict | list[dict], 
        st_makers_kwargs : dict | list[dict],
        relax_makers_kwargs : dict | list[dict],
        pmg_structures : Structure | list[Structure],
        idxs_structures : int | list[int],
        output_dir : str | None = None,
        ):
        """
        For each structure, run the phonon workflow in serial within the computing node.
        If multiple structures are provided, each structure's phonon workflow is run in serial, one after the other.
        """
        #Trasform kwargs and the input structure to a list of structures if it is not already
        if not isinstance(pmg_structures, list):
            pmg_structures = [pmg_structures]
        if not isinstance(ph_makers_kwargs, list):
            ph_makers_kwargs = [ph_makers_kwargs]
        if not isinstance(st_makers_kwargs, list):
            st_makers_kwargs = [st_makers_kwargs]
        if not isinstance(relax_makers_kwargs, list):
            relax_makers_kwargs = [relax_makers_kwargs]
        if not isinstance(idxs_structures, list):
            idxs_structures = [idxs_structures]
        
        #Assert same specified kwargs for PhononMakers
        if not (len(ph_makers_kwargs) == len(st_makers_kwargs) == len(relax_makers_kwargs)):
            raise ValueError("The input kwargs lists for PhononMakers must have the same length.")
        
        #Assert that the structures and idxs are lists of the same length
        if not (len(pmg_structures) == len(idxs_structures)):
            raise ValueError("The input structures and idxs lists must have the same length. Please provide the same number of structures and indices.")

        #If mulitple structures but only one set of kwargs is provided, replicate the kwargs for each structure
        if len(ph_makers_kwargs) == 1 and len(pmg_structures) > 1:
            # If only one set of kwargs is provided, replicate it for each structure
            ph_makers_kwargs = ph_makers_kwargs * len(pmg_structures)
            st_makers_kwargs = st_makers_kwargs * len(pmg_structures)
            relax_makers_kwargs = relax_makers_kwargs * len(pmg_structures)

        #Assert that the lists have the same length
        if not (len(ph_makers_kwargs) == len(pmg_structures)):
            raise ValueError("Structure list and PhononMakers kwargs lists must have the same length.")

        #Define local stores for docs and data
        # local_docs_store = JSONStore(paths="docs_store.json", read_only=False) #JSONStore seems not compatible with PhononMaker...
        # local_data_store = JSONStore(paths="data_store.json", read_only=False)
        local_docs_store = MemoryStore()
        local_data_store = MemoryStore()
        
        #Define Jobstore
        local_store = JobStore(
            docs_store=local_docs_store,
            additional_stores={"data": local_data_store}
            )
        
        #Create direcory for storing the results
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        #Loop over the structures and their corresponding makers
        stability_outputs = []
        for idx_structure, pmg_structure, ph_maker_kwargs, st_maker_kwargs, relax_maker_kwargs in zip(
            idxs_structures, pmg_structures, ph_makers_kwargs, st_makers_kwargs, relax_makers_kwargs
            ):

            #Create a ForceFieldStaticMaker instance
            static_maker = ForceFieldStaticMaker(**st_maker_kwargs)
        
            #Create a ForceFieldRelaxMaker instance
            relax_maker = ForceFieldRelaxMaker(**relax_maker_kwargs)
        
            # Create a PhononMaker instance
            phonon_maker = PhononMaker(**ph_maker_kwargs, 
                                    static_energy_maker=static_maker,
                                    bulk_relax_maker=relax_maker,
                                    phonon_displacement_maker=static_maker,
                                    )    

            # Create a phonon flow for the current structure
            ph_flow = phonon_maker.make(pmg_structure)

            #Map job name <-> uuid, index
            name_to_uuid = {job.name: job.uuid for job in ph_flow.jobs}
            name_to_idx = {job.name: job.index for job in ph_flow.jobs}
            print(f"job name <-> uuid: {name_to_uuid}") ##DEBUG
            print(f"job name <-> index: {name_to_idx}")

            # Run the flow locally
            response_dict = run_locally(
                flow=ph_flow,
                log="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                store=local_store, 
                create_folders=False, 
                # ensure_success=True,
                )

            #Map phonon jobs to their uuids and indices
            bulk_uuid, bulk_index = name_to_uuid["Force field relax"], name_to_idx["Force field relax"]
            gen_uuid, gen_index = name_to_uuid["generate_frequencies_eigenvectors"], name_to_idx["generate_frequencies_eigenvectors"]
            
            try:
                #Retrieve outputs: structure, phonon DOS and bandstructure
                relaxed_structure = response_dict[bulk_uuid][bulk_index].output.structure
                phonon_dos = response_dict[gen_uuid][gen_index].output.phonon_dos
                phonon_bs = response_dict[gen_uuid][gen_index].output.phonon_bandstructure
            except Exception as e:
                print(f"Error retrieving outputs for structure {idx_structure}: {e}")
                relaxed_structure, phonon_bs, phonon_dos = None, None, None

            #Compute dynamic stability fom
            if phonon_dos is not None:
                pos_modes, neg_modes = self._compute_dynamic_stability(phonon_dos)
            else:
                pos_modes, neg_modes = -1, -1

            #Get structure's output
            pmodes_ratio = (pos_modes, neg_modes)
            stability_output = {
                "structure_index": idx_structure,
                "relaxed_structure": relaxed_structure,
                "phonon_dos": phonon_dos, # do not save dos to save up space
                "phonon_bandstructure": phonon_bs, #do not save bs to save up space
                "pmodes_ratio": pmodes_ratio,
            }

            #Dump stability output: this method simply move the .yaml outputs to the specified output directory
            if output_dir is not None:
                stability_output_dir = f"{output_dir}/structure{idx_structure}/"
                self._dump_stability_output(
                    output_path=stability_output_dir,
                    # stability_output=stability_output,
                    save_bandstructure=True,  # Set to True to save phonon bandstructure
                    )
            
            #Remove the phonon DOS and bandstructure from the stability output to save up space in the database
            stability_output.pop("phonon_dos", None), stability_output.pop("phonon_bandstructure", None)
            stability_outputs.append(stability_output)

        #List with one dictionary for each structure, containing the stability outputs: 
        # structure index, relaxed structure, phonon DOS, phonon bandstructure and positive/negative modes ratios
        return stability_outputs

    @job
    def compute_stability_metrics(self, list_of_pstability_results, neg_modes_threshold=0.01):
        """
        """

        # Concatenate the list of stability results into a single list
        pstability_results = []
        for stability_result in list_of_pstability_results:
            pstability_results.extend(stability_result) #List of stability outputs, one for each structure

        # Collect the indeces of the structures that have negative and positive modes
        pos_modes_indices, neg_modes_indices = [], []
        final_structures, order_structures = [], []
        for pstability_result in pstability_results:
            #Get structure index
            structure_id = pstability_result["structure_index"]

            # Get the pos/neg phonon mods ratio from the structure
            pos_ratio, neg_ratio = pstability_result["pmodes_ratio"]

            # Perform a check on the ratio of negative modes
            if neg_ratio > neg_modes_threshold:
                neg_modes_indices.append(structure_id)
            elif 0 <= neg_ratio <= neg_modes_threshold:
                pos_modes_indices.append(structure_id)
            else:
                raise ValueError(f"Negative modes ratio {neg_ratio} is not valid for structure {structure_id}")

            #Get ase Atoms object from the pymatgen structure
            ase_atoms = AseAtomsAdaptor.get_atoms(pstability_result["relaxed_structure"])
            
            # Save final relaxed structure
            final_structures.append(ase_atoms)
            order_structures.append(structure_id)

        # Re-order the final structures according to the original structure indices
        final_structures = [final_structures[i] for i in order_structures]

        # Dump final relaxed structures
        relaxed_structure_fname = os.getcwd() + "/relaxed_structures.extxyz"
        write(relaxed_structure_fname, final_structures)

        # Save the indices of positive and negative modes to files
        stable_idxs_fname, unstable_idxs_fname = os.getcwd() + "/stable_structure_indices.txt", os.getcwd() + "/unstable_structure_indices.txt"
        np.savetxt("stable_structure_indices.txt", pos_modes_indices, fmt="%d", comments=f"# Structures with negative modes ratio < {neg_modes_threshold}")
        np.savetxt("unstable_structure_indices.txt", neg_modes_indices, fmt="%d", comments=f"# Structures with negative modes ratio > {neg_modes_threshold}")
        
        # Print the number of stable and unstable structures
        print(f"Found {len(pos_modes_indices)} stable structures and {len(neg_modes_indices)} unstable structures")
        print(f"Using negative modes threshold of {neg_modes_threshold}")

        return relaxed_structure_fname, stable_idxs_fname, unstable_idxs_fname

    def _compute_dynamic_stability(self,
        phonon_dos : PhononDos,
        ):
        """
        Compute figure of merit for the dynamic stability of structures
        by integrating the positive and negative phonon mode densities.
        """
        #Get frequencies and densities
        frequencies, densities = phonon_dos.frequencies, phonon_dos.densities

        #Integrate positive and negative densities
        number_positive_modes = simpson(y=densities[frequencies > 0], x=frequencies[frequencies > 0])
        number_positive_modes = abs(number_positive_modes)
        number_negative_modes = simpson(y=densities[frequencies < 0], x=frequencies[frequencies < 0])
        number_negative_modes = abs(number_negative_modes)
        total_number_modes = simpson(y=densities, x=frequencies)

        #Compute ratio of modes
        ratio_positive_modes = number_positive_modes / total_number_modes if total_number_modes != 0 else -1
        ratio_negative_modes = number_negative_modes / total_number_modes if total_number_modes != 0 else -1

        return ratio_positive_modes, ratio_negative_modes

    def _dump_stability_output(self,
        output_path : str,
        save_bandstructure : bool = False,
        ):
        """
        Move the output files from the current directory to the specified output directory.
        If save_bandstructure is True, the phonon bandstructure will be moved as well.
        """
        #Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        #Get working directory
        cwd = os.getcwd()

        #Move phonopy.yaml file
        os.rename(f"{cwd}/phonopy.yaml", f"{output_path}/phonopy.yaml")

        #Move phonon DOS file
        os.rename(f"{cwd}/phonon_dos.yaml", f"{output_path}/phonon_dos.yaml")

        #Move phonon bandstructure file if requested
        if save_bandstructure:
            os.rename(f"{cwd}/phonon_band_structure.yaml", f"{output_path}/phonon_band_structure.yaml")