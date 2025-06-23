import os
import argparse
from glob import glob


def clean_qe_folder(folder_path, remove_pwo=False):
    """
    Cleans up the Quantum ESPRESSO folder by removing un-finished pwo files and restoring their corresponding pwi file.

    Args:
        folder_path (str): Path to the Quantum ESPRESSO folder to be cleaned.
    """

    # Get pwo files
    pwo_files = glob(os.path.join(folder_path, "*.pwo"))
    
    #Loop through pwo files and check them
    for pwo_file in pwo_files:
        #Get condition to remove
        to_remove = True

        #Get corresponding pwi file
        pwi_file = pwo_file.replace(".pwo", ".pwi*")
        pwi = glob(pwi_file)[0]

        #Read pwo lines
        with open(pwo_file, 'r') as file:
            pwo_lines = file.readlines()

        #Check if pwo file is finished
        for line in pwo_lines:
            if "JOB DONE" in line:
                print(f"Found finished job in {pwo_file}")
                to_remove = False
        
        #Remove pwo file if not finished
        if to_remove:
            #Restore pwi file to be re-processed
            print(f"Renaming corresponding pwi file: {pwi}")
            basename_pwo = os.path.basename(pwo_file)
            restore_pwi_fname = os.path.dirname(pwi) + "/" + basename_pwo.replace(".pwo", ".pwi")
            os.rename(pwi, restore_pwi_fname)

            #Remove un-finished pwo file
            print(f"Unfinished job file: {pwo_file}")
            os.rename(pwo_file, pwo_file + "_to_be_removed")
            if remove_pwo:
                print(f"Removing file: {pwo_file}")
                # os.remove(pwo_file + "_to_be_removed")


#Get command line arguments
if __name__ == "__main__":
    #Define command line arguments
    parser = argparse.ArgumentParser(description="Clean Quantum ESPRESSO folder by removing unfinished pwo files.")
    parser.add_argument("folder_path", type=str, help="Path to the Quantum ESPRESSO folder to be cleaned.")
    parser.add_argument("--remove", required=False, action="store_true", default=False, help="If set, remove the pwo files instead of renaming them.")
    
    #Parse arguments
    args = parser.parse_args()
    qe_folder = args.folder_path
    remove_files = args.remove
    
    #Call the clean function
    clean_qe_folder(qe_folder, remove_pwo=remove_files)