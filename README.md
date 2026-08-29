# MACEDefect (fused into mace)

Charge-aware MACE with charge equilibration (QEq) for charged point defects in periodic solids, fused directly into this mace codebase as model type `MACEDefect` — no separate package.

`E_total = E0 + E_inter + E_QEq` — the total charge `Q` conditions a learned embedding added to every atom's species embedding before message passing, and QEq charges/energy are solved analytically each forward pass.

Everything else (training CLI, `MACECalculator`, cuEquivariance acceleration, checkpoints) works exactly the same way as any other mace model — see the main [README.md](README.md) for those.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training

```bash
mace_run_train --config config/scratch_config.yaml
```

`--model MACEDefect` is what selects this architecture; charge states are auto-detected from `total_charge` in the training data, and E0s are computed from neutral (Q=0) structures only.

### Energy & forces

```python
import numpy as np
from mace.calculators import MACECalculator
from ase.io import read, write

atoms = read("structure.vasp", format="vasp")
atoms.info["total_charge"] = -1

calc = MACECalculator(model_paths="checkpoint.model", model_type="MACE-Defect", device="cuda")
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
interaction_energy = calc.results["interaction_energy"]
qeq_energy = calc.results["qeq_energy"]

# per-atom QEq charges, written alongside the structure
atoms.arrays["charge"] = np.array(calc.results["qeq_charges"]).flatten()
write("structure_charges.xyz", atoms, format="extxyz")
```

### Relaxation

```python
import numpy as np
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
from ase.io import read, write

atoms = read("structure.vasp", format="vasp")
atoms.info["total_charge"] = -1

calc = MACECalculator(model_paths="checkpoint.model", model_type="MACE-Defect", device="cuda")
atoms.calc = calc

opt = LBFGS(atoms, logfile="relax.log")
opt.run(fmax=0.01)

print(f"Relaxed energy: {atoms.get_potential_energy():.4f} eV")

# per-atom QEq charges, written alongside the relaxed structure
atoms.arrays["charge"] = np.array(calc.results["qeq_charges"]).flatten()
write("relaxed_charges.xyz", atoms, format="extxyz")
```
