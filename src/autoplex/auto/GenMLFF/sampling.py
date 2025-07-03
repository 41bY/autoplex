import os
from typing import Literal
from dataclasses import dataclass, field

import numpy as np
from jobflow import Maker

from ase import Atoms
from ase.io import read, write
from dscribe.descriptors import SOAP
from skmatter.sample_selection import FPS, CUR


@dataclass
class SamplingMaker(Maker):
    """
    Make class to sample structures based on the specified method.
    Parameters
    ----------
    name: str
    Name of the Maker. Default is "atomic_structures_sampler".

    structure_path: str | None
    Path to the file containing structures. Structures path must be provided.

    num_of_selection: int
    Number of structures to be sampled. Default is 5.

    selection_method : Literal["cur", "fps", "bfps1s", "bcur1s", "bcur2i", "bfps2i", "random", "uniform"]
    Method for selecting samples. Options include:
        - "cur": Pure CUR sampling.

        - "fps": Pure furthest point sampling (FPS).

        - "bcur": Boltzmann flat histogram in enthalpy, then CUR.
            - "bcur1s": Execute bcur with one shot (1s)
            - "bcur1s": Execute bcur with two iterations (2i)
        
        - "bfps": Boltzmann flat histogram in enthalpy, then FPS.
            - "bfps1s": Execute bfps with one shot (1s)
            - "bfps2i": Execute bfps with two iterations (2i)

        - "random": Random selection.

        - "uniform": Uniform selection.

    soap_params: dict
    SOAP descriptor parameters
        - 'l_max': int, Maximum degree of spherical harmonics (default 12).
        - 'n_max': int, Maximum number of radial basis functions (default 12).
        - 'sigma': float, Width of Gaussian smearing (default 0.0875).
        - 'r_cut: float, Radial cutoff distance (default 10.5).
        - 'periodic': bool, Whether to use periodic boundary conditions (default True).
        - 'average': ["inner", "outer"], Average method for SOAP vectors (default "outer").
        - 'species': bool, Whether to consider species information (default True).

    boltz_params: dict
    Boltzmann hentalpy probability parameters
        - 'kt': float, Temperature in eV for Boltzmann weighting (default 0.3).
        - 'boltz_frac': float, Fraction of Boltzmann CUR selections (default 0.8).
        - 'bolt_max_num': int, Maximum number of Boltzmann selections (default 3000).
    
    isolated_atom_energies: dict
        Dictionary of isolated energy values for species. Required for 'boltzhist_cur'
        selection method. Default is None.
    
    random_seed: int, optional
        Seed for random number generation, ensuring reproducibility of sampling.

    Returns
    -------
    list of ase.Atoms
        The selected atoms.
    """    
    name: str = "atomic_configuration_sampling"
    structure_path: str | None = None  # Path to the file containing structures
    num_of_samples: int = 5 # Number of structures to sample
    selection_method: Literal["cur", "bcur1s", "bcur2i", "fps", "bfps1s", "bfps2i", "random", "uniform"] | None = None # Method for selecting samples
    soap_params: dict = field(default_factory=dict)  # SOAP descriptor parameters
    boltz_params: dict = field(default_factory=dict)  # Boltzmann hentalpy weighting parameters
    isolated_atom_energies: dict | None = None
    random_seed: int = None

    def make(self) -> str:
        """
        Main maker method to sample structures based on the specified method.

        Returns
        -------
        str
            Path to extxyz file with the sampled ase.Atoms.
        """
        #Generate random seed if not provided
        if self.random_seed is None:
            self.random_seed = np.random.randint(0, 10000)

        #Read structures from file
        structures = read(self.structure_path, index=":")     
        
        #Perform selection based on the specified method
        if self.selection_method == "random":
            selected_atoms = self.random_selection(atoms=structures)
        
        elif self.selection_method == "uniform":
            selected_atoms = self.uniform_selection(atoms=structures)
        
        elif self.selection_method == "cur":
            selected_atoms = self.descriptor_based_selection(atoms=structures, method="cur")

        elif self.selection_method == "fps":            
            selected_atoms = self.descriptor_based_selection(atoms=structures, method="fps")

        elif self.selection_method == "bcur1s":
            selected_atoms = self.boltz_and_descriptor_selection(atoms=structures, descriptor_method="cur")

        elif self.selection_method == "bfps1s":
            selected_atoms = self.boltz_and_descriptor_selection(atoms=structures, descriptor_method="fps")
        
        elif self.selection_method == "bcur2i":
            # Perform Boltzmann and descriptor selection with two iterations
            first_selected_atoms = self.boltz_and_descriptor_selection(atoms=structures, descriptor_method="cur")
            selected_atoms = self.boltz_and_descriptor_selection(atoms=first_selected_atoms, descriptor_method="cur")
        
        elif self.selection_method == "bfps2i":
            # Perform Boltzmann and descriptor selection with two iterations
            first_selected_atoms = self.boltz_and_descriptor_selection(atoms=structures, descriptor_method="fps")
            selected_atoms = self.boltz_and_descriptor_selection(atoms=first_selected_atoms, descriptor_method="fps")            
        
        else:
            raise ValueError(
                f"Invalid selection method: {self.selection_method}. "
                "Please choose from 'cur', 'bcur1s', 'bcur2i', 'fps', 'bfps1s', 'bfps2i', 'random', or 'uniform'."
            )
        
        #Dump selected atoms to file
        selected_structure_path = os.getcwd() + f"/sampled_structures.extxyz"
        write(selected_structure_path, selected_atoms, format="extxyz")

        return selected_structure_path

    def random_selection(self, atoms) -> list[Atoms]:
        """
        Select structures randomly from the provided structures.

        Returns
        -------
        list[Atoms]
            List of randomly selected Atoms objects.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        if len(atoms) < self.num_of_samples:
            selected_atoms = atoms
        else:
            selected_idxs = np.random.choice(len(atoms), self.num_of_samples, replace=False)
            selected_atoms = [atoms[i] for i in selected_idxs]

        return selected_atoms

    def uniform_selection(self, atoms) -> list[Atoms]:
        """
        Select structures uniformly from the provided structures.

        Returns
        -------
        list[Atoms]
            List of uniformly selected Atoms objects.
        """
        if len(atoms) < self.num_of_samples:
            selected_atoms = atoms
        else:
            indices = np.linspace(0, len(atoms) - 1, self.num_of_samples, dtype=int)
            selected_atoms = [atoms[idx] for idx in indices]

        return selected_atoms

    def descriptor_based_selection(self, atoms, descriptors=None, method=None) -> list[Atoms]:
        """
        Select structures using CUR (Curated Uncertainty Reduction) method.

        Returns
        -------
        list[Atoms]
            List of selected Atoms objects using CUR method.
        """
        #Check that number of samples is not greater than number of structures
        if len(atoms) < self.num_of_samples:
            print(f"Number structures to be selected ({self.num_of_samples}) exceeds number of available structures ({len(atoms)}): selecting all structures.")
            return atoms

        # Compute SOAP descriptor for the given structures
        if descriptors is None:
            descriptors = self._compute_soap_descriptors(atoms)

        #Choose the selector based on the specified method
        if method == "cur":
            #CUR selector
            selector = CUR(
                n_to_select=self.num_of_samples,
                random_state=self.random_seed,
            )                    
        
        elif method == "fps":
            #FPS selector
            selector = FPS(
                n_to_select=self.num_of_samples,
                random_state=self.random_seed,
                initialize='random'
            )
        
        else:
            raise ValueError(f"Invalid selection method: {method}. Please choose 'cur' or 'fps'.")

        # Select structures
        selector.fit(descriptors)
        selected_atoms = [atoms[i] for i in selector.selected_idx_]

        print("Descriptor-based selection:", method) ##DEBUG

        return selected_atoms

    def _compute_soap_descriptors(self, atoms):
        """
        Create SOAP descriptor for the provided structures.
        """
        #Get atomic species from structures
        species = set()
        for frame in atoms:
            species.update(frame.get_chemical_symbols())

        #Get parameters for soap descriptor
        default_soap_params = {
            "species": species,
            "r_cut": 10.0,
            "n_max": 12,
            "l_max": 12,
            "sigma": 1.0,
            "periodic": True,
            "average": "outer",
        }
        default_soap_params.update(self.soap_params)

        # Create SOAP descriptor: with 'outer' average we create a single descriptor for each structure
        soap = SOAP(**default_soap_params)

        # Compute the descriptor for each structure dimensions = (n_frames, n_features)
        descriptors = soap.create(atoms, n_jobs=-1) #Parallel computation of descriptors

        return descriptors

    def _boltzmann_selection(self, atoms):
        """
        Select structures using Boltzmann histogram in enthalpy, adapted from autoplex-rss.
        """               
        #Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        #Get parameters for boltzmann average
        default_boltz_params = {
            "kt": 0.3,
            "boltz_frac": 0.8,
            "bolt_max_num": 3000,
        }
        default_boltz_params.update(self.boltz_params)

        # Get number of structures to select with boltzmann selection
        select_num = round(default_boltz_params["boltz_frac"] * len(atoms))
        select_num = select_num if select_num < default_boltz_params["bolt_max_num"] else default_boltz_params["bolt_max_num"]

        #Formation energy calculation
        #Get isolated atomic energies #eV
        e0s = {int(k): float(v) for k, v in self.isolated_atom_energies.items()}    
        #Compute FEs #eV
        formation_energies = [frame.get_potential_energy() - sum(e0s[z] for z in frame.get_atomic_numbers()) for frame in atoms]

        #Compute pressures #eV/Å³
        pressures = [frame.get_stress(voigt=True)[:3].mean() for frame in atoms] 

        #Compute enthalpies
        enthalpies = [fe + frame.get_volume() * p for fe, frame, p in zip(formation_energies, atoms, pressures)]
        enthalpies = np.array(enthalpies)
        print("Enthalpies for Boltzmann selection:", enthalpies) ##DEBUG

        #Compute frame's relative probabilities
        min_H = np.min(enthalpies) # Most stable configuration
        histo = np.histogram(enthalpies)
        kt, config_prob = default_boltz_params["kt"], []
        for H in enthalpies:
            bin_i = np.searchsorted(histo[1][1:], H, side="right")
            if bin_i == len(histo[1][1:]): #Greatest entalpy
                bin_i = bin_i - 1
            
            #Compute relative probability
            p = 1.0 / histo[0][bin_i] if histo[0][bin_i] > 0.0 else 0.0
            if kt > 0.0:
                p *= np.exp(-(H - min_H) / kt)
            config_prob.append(p)
        config_prob = np.array(config_prob)
        
        #Perform Boltzmann sampling
        selected_bolt_atoms = []
        for _ in range(select_num):
            # Compute probability distribution from relative probabilities
            config_prob /= np.sum(config_prob)
            # Compute cumulative probabilities
            cumul_prob = np.cumsum(config_prob)
            # Sample the cumulative probabilities
            rv = np.random.uniform()
            config_i = np.searchsorted(cumul_prob, rv)
            selected_bolt_atoms.append(atoms[config_i])
            # Remove the selected configuration from the lists
            config_prob = np.delete(config_prob, config_i)
            del atoms[config_i]
            enthalpies = np.delete(enthalpies, config_i)

        return selected_bolt_atoms

    def boltz_and_descriptor_selection(self, atoms, descriptors=None, descriptor_method=None):
        """
        Perform Boltzmann selection followed by descriptor-based selection.
        """
        # First, perform Boltzmann selection
        boltz_selected_atoms = self._boltzmann_selection(atoms=atoms)

        # Then, perform descriptor-based selection on the Boltzmann selected atoms
        selected_atoms = self.descriptor_based_selection(atoms=boltz_selected_atoms, descriptors=descriptors, method=descriptor_method)

        return selected_atoms