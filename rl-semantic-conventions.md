# RL semantic conventions

The specification is maintained in the independent `rl-lens` repository at
`src/rl_lens/rl-semantic-conventions.md` and is included in the `rl-lens` wheel.

After installing the package, locate the bundled specification with:

```python
from importlib.resources import files

specification = files("rl_lens").joinpath("rl-semantic-conventions.md")
```
