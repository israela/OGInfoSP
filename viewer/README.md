# Simulation Results Viewer

The interactive viewer displays the saved Gaussian and CIFAR-10 simulation results included with this repository.

## Setup

The viewer requirements support Python 3.10 or newer.

1. Clone the repository as described in the [main README](../README.md).
2. From the repository root, install the viewer packages:

```powershell
python -m pip install -r requirements-viewer.txt
```

3. Extract `simulation-results-July-2026.zip` into the repository root. After extraction, the results should be under `simulation-results-July-2026/July_2026`. The viewer also supports results placed directly under `output/July_2026`.

### Troubleshooting long paths on Windows

If extraction fails because a file path is too long, move the repository to a shorter path, such as `C:\OGInfoSP`, and extract the ZIP there. Alternatively, assign the repository folder a temporary drive letter (replace the example path as needed):

```powershell
subst O: "C:\path\to\OGInfoSP"
```

Open `O:\` in File Explorer and extract the ZIP there. When finished, remove the temporary drive letter:

```powershell
subst O: /d
```

## Launch

On Windows, double-click `viewer/run_viewer.bat`. The viewer will open in your web browser.

Alternatively, run the following command from the repository root:

```powershell
streamlit run viewer/viewer.py
```
