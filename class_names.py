import os
import sys
from collections import defaultdict

def extract_class_names(data_dir, output_file="pet_breed_names.txt"):
    """
    Extract all unique class names from the Oxford-IIIT Pet Dataset.
    Returns a dictionary mapping each class name to an example filename.
    Outputs results to a text file.
    """
    class_examples = {}
    all_breeds = defaultdict(int)
    cats = []
    dogs = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        sys.exit(1)
    
    # Loop through all files
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.jpg'):
            # Extract the breed name (everything before the underscore)
            parts = filename.split("_")
            if len(parts) < 2:
                continue
                
            breed_name = parts[0]
            
            # If we haven't seen this breed before, add it to our collection
            if breed_name not in class_examples:
                class_examples[breed_name] = filename
                all_breeds[breed_name] += 1
                
                # Categorize as cat or dog based on capitalization
                if breed_name[0].isupper():
                    cats.append(breed_name)
                else:
                    dogs.append(breed_name)
            else:
                all_breeds[breed_name] += 1
    
    # Prepare output
    output = []
    output.append(f"Found {len(class_examples)} unique breeds:")
    output.append(f"- {len(cats)} cat breeds (uppercase first letter)")
    output.append(f"- {len(dogs)} dog breeds (lowercase first letter)")
    
    output.append("\n--- Cat Breeds ---")
    for breed in sorted(cats):
        output.append(f"{breed} (Example: {class_examples[breed]}, Count: {all_breeds[breed]})")
    
    output.append("\n--- Dog Breeds ---")
    for breed in sorted(dogs):
        output.append(f"{breed} (Example: {class_examples[breed]}, Count: {all_breeds[breed]})")
        
    output.append("\n--- List of Breeds (for config file) ---")
    output.append(str(cats + dogs))
    
    output.append("\n--- For Few-Shot Holdout Configuration ---")
    output.append("If your FEW_SHOT_HOLDOUT_BREEDS list includes these breeds:")
    output.append("['american_bulldog', 'Egyptian_Mau', 'samoyed', 'Siamese', 'wheaten_terrier']")
    output.append("\nMake sure they match exactly with names in this file!")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))
        
    # Print to console
    for line in output:
        print(line)
    
    print(f"\nResults have been saved to {output_file}")
    return class_examples

if __name__ == "__main__":
    # Default path from your example, adjust as needed
    data_dir = "data/archive/images"
    
    # Allow command line override
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    extract_class_names(data_dir)