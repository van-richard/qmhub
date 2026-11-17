help([[
QMHub from the tested vibing source, installed in the lynnx qmhub2 environment.

Initialize Lmod in Bash before using this Lua module:
    source /opt/apps/lmod/lmod/8.7.34/init/bash
    module load qmhub2

Deactivate any Conda environment and unload other environment modules first.
This module selects executables through PATH; it does not activate Conda or
change backend modules, library search paths, or thread settings. Sander and
Q-Chem remain separate selections. Unloading restores Python search settings.
]])

whatis("QMHub vibing: lynnx qmhub2 executable selector")

-- Explicit conflicts fail instead of silently swapping environment families.
conflict("qmhub2-mlpui", "qmhub", "base", "ambertools23", "mbar", "webdev", "conda")

local prefix = "/home/van/Programs/miniforge3/envs/qmhub2"
if mode() == "load" then
    if (os.getenv("CONDA_PREFIX") or "") ~= "" then
        LmodError("Deactivate the active Conda environment before loading qmhub2.")
    end
    if not isFile(pathJoin(prefix, "bin", "python")) then
        LmodError("The qmhub2 Python environment is missing: " .. prefix)
    end
end

-- Avoid Conda activation hooks and global library overrides that can affect Amber.
prepend_path("PATH", pathJoin(prefix, "bin"))
pushenv("PYTHONNOUSERSITE", "1")
-- false unsets PYTHONPATH while retaining its previous value for module unload.
pushenv("PYTHONPATH", false)
