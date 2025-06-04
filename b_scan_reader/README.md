# B Scan Exporter

## Installation Instructions
1. Install Python 3.9.10
    - Download and run [Python 3.9.10](https://www.python.org/downloads/release/python-3910/) (internet connection is required)
    - Ensure the "Add Python 3.9 to PATH" checkmark is checked
    - Press "Install Now"
    - Once installation finishes, select "Disable path length limit" on proceeding pop-up
2. Install Poetry
   -  Enter the command `pip install poetry`
3. Install Python Dependencies
    - Navigate to code base root directory in file explorer
    - In the address bar of windows explorer, type `cmd`. This will open up a command prompt in the projects root directory
    - Enter the command `poetry install`
      - If you run into issues installing the poetry virtual envirionment, see the section [Poetry Installation Troubleshooting](#poetry-installation-troubleshooting).

### Poetry Installation Troubleshooting
If you have [multiple versions of python installed](https://python-poetry.org/docs/managing-environments/), poetry may not be able to find the desired python version.

If you get the error `The currently activated Python version _._._ is not supported by the project (^3.9).` follow the below steps to resolve it. In these steps we will tell poetry which python version to use for the project, and that will allow us to install the dependencies.

1. Locate your `python 3.9.10` install directory. 
   - Open a command prompt and type `python` or `py` or whichever command is tied to `python 3.9.10`.
   - Type the following commands  
   ```
   import sys
   sys.executable
   exit()
   
   ```
   This will return a line similar to:
   `'C:\\Users\\<user>\\AppData\\Local\\Programs\\Python\\Python39\\python.exe'`
2. Copy the path printed in the step above. Replace the double-backslashes with single slashes. Replace the single quotes with double quotes (not sure why it needs double quotes, but it wouldn't install for me with single quotes)
3. Type `poetry env use "<full\path\to\python.exe>"`
   - Where `"<full\path\to\python.exe>"` is the path printed in Step 1
4. Type `poetry install`



### Using Poetry
- Use `poetry run <command>` to run individual commands in the terminal.
- Use `poetry shell` to activate the poetry environment.
  - When in a shell, all commands will use the poetry venv. You don't have to prepend commands with `poetry run`


## Exporting B-Scans to .csv files
1. Navigate to the project directory with windows explorer and tyoe `cmd` in the address bar
2. Type the following command in the command prompt:

`poetry run python export_bscan.py "<b-scan *.anf or *daq file path>" "<output csv folder location>"`

where, 

`<b-scan *.anf or *.daq file path>` is the file path location of the .anf or .daq bscan file and 

`<output csv folder location>` is the desired output location of the output csv files. For example:

`poetry run python export_bscan.py "C:\Users\plt\Desktop\BSCAN Type D  B-07 Pickering B Unit-6 west 09-Feb-2018 060443 [A2370-2503][R0-3599].anf" "C:\Users\plt\Desktop\New folder"`

or

`poetry run python export_bscan.py "C:\Users\plt\Desktop\DAQ P0 Z00E 2021-Nov-06 125018 [A3242-3499].daq" "C:\Users\plt\Desktop\New folder"`
f
NOTE: 
- This process may take 1-2 minutes per probe. The .daq files take longer to export as they use u16 data, rather than u8 in the .anf files.
- Ensure file paths have quotations around them
- Ensure the `<output csv folder location>` path is a folder, not a file

## Testing
Run the command `pytest` in the `b_scan_reader` root directory

    
