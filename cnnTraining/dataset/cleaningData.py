import os
import csv

def swap_csv_columns(input_folder, output_folder):
    """
    Goes through all CSV files in the input_folder, swaps the 2nd and 3rd columns,
    and saves the modified files into the output_folder.
    """
    # Create the output folder if it doesn't exist yet
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            # Open the input file for reading and the output file for writing
            with open(input_filepath, mode='r', newline='', encoding='utf-8') as infile, \
                 open(output_filepath, mode='w', newline='', encoding='utf-8') as outfile:
                
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                # Process each row in the CSV
                for row in reader:
                    # Ensure the row actually has at least 3 columns before trying to swap
                    if len(row) >= 3:
                        # Swap the 2nd column (index 1) and 3rd column (index 2)
                        row[1], row[2] = row[2], row[1]
                    
                    # Write the modified (or untouched) row to the new file
                    writer.writerow(row)
            
            print(f"Processed: {filename}")

    print("\nAll done! Your corrected files are in the output folder.")

# --- How to use ---
# Replace these paths with the actual paths on your computer
INPUT_DIR = './mislabeled_jumping'   # Folder where your current files are
OUTPUT_DIR = './jumping'     # Folder where you want the new files to go

swap_csv_columns(INPUT_DIR, OUTPUT_DIR)