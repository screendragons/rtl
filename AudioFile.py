import pandas as pd
import os

class AudioFile:
    def remove_non_existing_audio_files(audio_files):
        existing_files = []
        for file_path in audio_files:
            if not os.path.exists(file_path):
                print("File not found: " + file_path + ". Skipping...")
            else:
                existing_files.append(file_path)
                
        return existing_files
    
    # SUBGROUPS:
    # GROUP 1: native children aged 7-11 (DC)
    # GROUP 2: native children aged 12-16 (DT)
    # GROUP 3: non-native children (NNC)
    # GROUP 4: non-native adults (NNA)
    # GROUP 5: native adults above 65 (DOA)

    def get_audio_files_by_subgroup(metadata_txt_path, audio_files_path, subgroup, component="comp-q"):
        # read text file into pandas DataFrame
        metadata = pd.read_csv(metadata_txt_path, sep='\t')

        # Get all records for the given component
        metadata = metadata[metadata.Component == component]

        # Get all records for the given subgroup
        metadata = metadata[metadata.Group == subgroup]
       
        # Convert the dataframe to a list of files
        audio_files = list(metadata["Root"])
        
        # For every file, prepend the full path and append the file extension (.wav)
        audio_files = [audio_files_path + file + ".wav" for file in audio_files]

        # Remove audio files that do not exist
        audio_files = AudioFile.remove_non_existing_audio_files(audio_files)

        return audio_files
    

    def get_metadata_by_nr(root):
        # read text file into pandas DataFrame
        metadata = pd.read_csv(os.getcwd() + "/files/metadata/recordings_NL.txt", sep='\t')

        # Get all records for the given component
        metadata = metadata[metadata.Root == root]
        return metadata

