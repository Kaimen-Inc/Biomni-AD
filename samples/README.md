# Sample input

Biomni-AD needs no sample input: it works on whatever is in the user's workspace, and on the public AD data lake, which it downloads as questions need it.

`synthetic_gwas_hits.csv` exists to check a new deployment end to end.
Its gene and chromosome columns are real, but its p-values are invented, so it is only fit for testing.

Copy it into a workspace, open Biomni-AD from the platform, and ask:

> Plot the -log10 p-values in synthetic_gwas_hits.csv as a bar chart.

Approve the plan it proposes.
A working deployment then reads the file from the workspace, runs the plotting code, shows the chart, and saves it with the code that drew it in a new run folder under the output directory.
Each step exercises one part of the deployment: the workspace mount, the model calls through the LLM proxy, and the output volume.
See [docs/grip_deployment.md](../docs/grip_deployment.md#smoke-test) for the full check.
