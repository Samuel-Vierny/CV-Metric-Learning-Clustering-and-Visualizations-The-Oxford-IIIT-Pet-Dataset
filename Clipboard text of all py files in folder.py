# Open folder in cursor or VScode. run this script

import os
import pyperclip

def main():
    folder = '.'
    all_content = ""

    for filename in os.listdir(folder):
        if filename.endswith('.py'):
            # Skip this script itself if needed
            if filename == os.path.basename(__file__):
                continue
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                # Optionally add a header with the file name for clarity
                all_content += f"# ----- {filename} -----\n" + file_content + "\n\n"

    # Copy the concatenated content to the clipboard
    pyperclip.copy(all_content)
    print("All Python file content has been copied to your clipboard.")

if __name__ == '__main__':
    main()
