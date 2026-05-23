import glob
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from biomni.agent.a1 import A1
from biomni.agent.ad_data_downloader import download_ad_catalog_data

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.theme import Theme

    custom_theme = Theme(
        {
            "info": "dim cyan",
            "warning": "magenta",
            "danger": "bold red",
            "success": "bold green",
            "header": "bold cyan underline",
        }
    )
    console = Console(theme=custom_theme)
    print = console.print  # Override print
except ImportError:
    # Fallback if rich is not available (though we verified it is)
    pass


class AD1(A1):
    def __init__(self, download_ad_data: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ad_keywords = ["Alzheimer", "AD", "dementia", "MCI", "amyloid", "tau", "neurodegeneration", "cognition"]

        if download_ad_data:
            self._bulk_download_ad_data()

        self._enforce_local_data_priority()

    def _bulk_download_ad_data(self):
        """Perform bulk download of BiomniAD catalog data (<100MB)."""
        try:
            console.print(
                Panel(
                    "[header]🚀 Initiating Bulk BiomniAD Data Download[/header]\nScanning catalogs for and fetching files < 100MB to local data lake.",
                    border_style="cyan",
                )
            )
        except NameError:
            print("🚀 Initiating Bulk BiomniAD Data Download...")

        results = download_ad_catalog_data(self.data_lake_dir)

        # After download, sync descriptions and inform user
        self._sync_data_lake_descriptions()
        self._save_custom_data_index()

        summary_text = (
            f"✅ [success]Downloaded:[/success] {len(results['downloaded'])} file(s)\n"
            f"⏭️ [info]Already present:[/info] {len(results['already_present'])} file(s)\n"
            f"🚫 [warning]Skipped (too large):[/warning] {len(results['skipped_too_large'])} file(s)\n"
            f"❌ [danger]Failed:[/danger] {len(results['skipped_error'])} file(s)"
        )

        try:
            console.print(Panel(summary_text, title="[header]Download Summary[/header]", border_style="green"))
            if results["downloaded"]:
                console.print(f"📁 Files stored in: [bold]{os.path.join(self.data_lake_dir, 'biomniAD')}[/bold]")
            if results["skipped_error"]:
                console.print("\n[bold yellow]⚠️  Failed downloads (first 20):[/bold yellow]")
                for entry in results["skipped_error"][:20]:
                    console.print(f"  [red]•[/red] {entry}")
        except NameError:
            print(summary_text)
            if results["skipped_error"]:
                print("\n⚠️  Failed downloads (first 20):")
                for entry in results["skipped_error"][:20]:
                    print(f"  • {entry}")

    def _build_local_data_priority_instruction(self) -> str:
        """Build AD1 local-data-first policy block for the system prompt."""
        local_items = []
        if hasattr(self, "_get_data_lake_items"):
            local_items = self._get_data_lake_items()

        data_root_dir = getattr(self, "data_root_dir", None)
        data_root_items = []
        if hasattr(self, "_get_data_root_items"):
            data_root_items = self._get_data_root_items(max_depth=5)

        data_lake_dir = getattr(self, "data_lake_dir", "")
        ad_data_lake = os.path.join(data_lake_dir, "biomniAD")

        # Build enriched BiomniAD dataset map from catalog JSONs
        catalog_datasets = self._load_catalog_datasets()
        local_ad_datasets = self._build_local_ad_dataset_inventory(ad_data_lake, catalog_datasets)

        # Format local AD datasets section
        ad_lines = []
        for entry in local_ad_datasets:
            ad_lines.append(f"  [{entry['id']}] {entry['title']}")
            ad_lines.append(f"    Description: {entry['description']}")
            ad_lines.append(f"    Directory: {entry['dir']}")
            for f in entry["files"]:
                ad_lines.append(f"    - {f}")
        ad_section = "\n".join(ad_lines) if ad_lines else "  (none downloaded yet)"

        # Non-AD data lake files (generic Biomni data)
        non_ad = [i for i in local_items if not i.startswith("biomniAD/")]
        non_ad_preview = "\n".join(f"  - {i}" for i in non_ad[:20])
        if len(non_ad) > 20:
            non_ad_preview += f"\n  - ... and {len(non_ad) - 20} more"

        # Data root preview — prefer pre-computed inventory from Chainlit sidebar
        root_preview = ""
        _precomputed = getattr(self, "user_data_inventory", None)
        if _precomputed:
            root_preview = f"\n\nUSER DATA DIRECTORY ({data_root_dir}):\n{_precomputed}"
        elif data_root_dir and data_root_items:
            root_dirs = sorted({item.split("/")[0] for item in data_root_items if "/" in item})
            root_files = [item for item in data_root_items if "/" not in item]
            root_lines = []
            for d in root_dirs[:80]:
                sub_count = sum(1 for i in data_root_items if i.startswith(d + "/"))
                root_lines.append(f"  - 📁 {d}/ ({sub_count} file(s))")
            for f in root_files[:30]:
                root_lines.append(f"  - 📄 {f}")
            if len(root_dirs) > 80:
                root_lines.append(f"  - ... and {len(root_dirs) - 80} more directories")
            root_preview = f"\n\nUSER DATA DIRECTORY ({data_root_dir}):\n" + "\n".join(root_lines)
        elif data_root_dir:
            root_preview = f"\n\nUSER DATA DIRECTORY ({data_root_dir}): (empty or not mounted)"

        return f"""
### AD1_LOCAL_DATA_POLICY_START
## AD1 DATA PRIORITY RULES — ALWAYS FOLLOW IN ORDER

1. **LOCAL FILES FIRST** — Use files already on disk. Do NOT fetch data that is already present.
   - BiomniAD data lake: {ad_data_lake}
   - General data lake: {data_lake_dir}
   - User data dir: {data_root_dir or "not set"}

2. **BiomniAD CATALOG** — If a dataset is listed below without local files, use its catalog URI to fetch.
   Catalogs: biomni/know_how/resource/BiomniAD_Discovery.json, NIAGADS_datasets_with_files.json, SinaiADRD.json

3. **Web / literature** — Only after checking local and catalog sources.

4. **Never fabricate data.** If a file is missing, say so explicitly.

---
## LOCALLY AVAILABLE BiomniAD DATASETS ({len(local_ad_datasets)} datasets, ready to load directly)

{ad_section}

## GENERAL DATA LAKE FILES ({len(non_ad)} files)
{non_ad_preview or "  (none)"}
{root_preview}
### AD1_LOCAL_DATA_POLICY_END
""".strip()

    def _load_catalog_datasets(self) -> dict[str, dict]:
        """Load all BiomniAD catalog JSONs and return a dict keyed by dataset id."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resource_dir = os.path.join(current_dir, "..", "know_how", "resource")
        datasets: dict[str, dict] = {}
        for pat in ["BiomniAD*.json", "NIAGADS*.json", "SinaiADRD.json"]:
            for catalog_path in glob.glob(os.path.join(resource_dir, pat)):
                try:
                    with open(catalog_path) as f:
                        data = json.load(f)
                    for ds in data.get("datasets", []):
                        ds_id = ds.get("id")
                        if ds_id:
                            datasets[ds_id] = ds
                except Exception:
                    pass
        return datasets

    def _build_local_ad_dataset_inventory(self, ad_data_lake: str, catalog_datasets: dict) -> list[dict]:
        """Return a list of biomniAD datasets that have local files, enriched with catalog descriptions."""
        if not os.path.isdir(ad_data_lake):
            return []
        entries = []
        for ds_id in sorted(os.listdir(ad_data_lake)):
            ds_dir = os.path.join(ad_data_lake, ds_id)
            if not os.path.isdir(ds_dir):
                continue
            local_files = sorted(
                f
                for f in os.listdir(ds_dir)
                if not f.startswith(".")
                and os.path.isfile(os.path.join(ds_dir, f))
                and not any(f.lower().startswith(p) for p in ("readme", "read_me"))
            )
            if not local_files:
                continue
            cat = catalog_datasets.get(ds_id, {})
            title = cat.get("title") or ds_id
            # Truncate long description to 1 sentence
            desc = (cat.get("study_description") or "").split(".")[0].strip()
            if len(desc) > 160:
                desc = desc[:157] + "..."
            entries.append(
                {
                    "id": ds_id,
                    "title": title,
                    "description": desc or "No description available.",
                    "dir": ds_dir,
                    "files": local_files[:10] + (["..."] if len(local_files) > 10 else []),
                }
            )
        return entries

    def _enforce_local_data_priority(self) -> None:
        """Ensure local-data-first policy is always present even after prompt updates."""
        if not hasattr(self, "system_prompt") or not self.system_prompt:
            return

        policy_block = self._build_local_data_priority_instruction()
        self.system_prompt = re.sub(
            r"### AD1_LOCAL_DATA_POLICY_START.*?### AD1_LOCAL_DATA_POLICY_END\\n?",
            "",
            self.system_prompt,
            flags=re.DOTALL,
        ).strip()
        self.system_prompt = f"{policy_block}\n\n{self.system_prompt}"

    def configure(self, *args, **kwargs):
        """Configure AD1 and enforce local-data-first policy for all tasks."""
        super().configure(*args, **kwargs)
        self._enforce_local_data_priority()

    def update_system_prompt_with_selected_resources(self, selected_resources):
        """Update prompt and then re-apply AD1 global local-data-first policy."""
        super().update_system_prompt_with_selected_resources(selected_resources)
        self._enforce_local_data_priority()

    def go(self, prompt):
        """Execute the agent with the given prompt, injecting AD context if relevant."""

        # Initialize log/raw_log early so _save_run_artifacts always has them,
        # even when super().go() raises an exception.
        self.log = getattr(self, "log", [])
        self.raw_log = getattr(self, "raw_log", [])

        # 1. Setup run directory
        run_id = self._build_run_id(prompt)
        runs_root = os.path.abspath(os.path.join(os.getcwd(), "runs"))
        current_run_dir = os.path.join(runs_root, run_id)
        os.makedirs(current_run_dir, exist_ok=True)

        try:
            console.print(
                Panel(
                    f"[bold white]🚀 Starting AD1 Agent Run[/bold white]\n[dim]ID: {run_id}[/dim]",
                    title="[header]Biomni AD1[/header]",
                    border_style="cyan",
                )
            )
        except NameError:
            print(f"\n🚀 Starting AD1 Agent Run: {run_id}")

        # 2. Capture initial file state
        initial_files = self._get_all_files(os.getcwd())

        # Check for AD keywords
        is_ad_task = any(keyword.lower() in prompt.lower() for keyword in self.ad_keywords)

        if is_ad_task:
            try:
                console.print(
                    "\n[magenta]🧠 AD/Dementia task detected.[/magenta] Injecting specialized data sourcing protocols..."
                )
            except NameError:
                print("\n🧠 AD/Dementia task detected. Injecting specialized data sourcing protocols...")
            self._inject_ad_context()

        # 3. Run the agent
        try:
            super().go(prompt)
        except Exception as e:
            try:
                console.print(f"\n[danger]❌ Agent execution failed:[/danger] {e}")
            except NameError:
                print(f"\n❌ Agent execution failed: {e}")

        # 4. Save artifacts
        self._save_run_artifacts(run_id, current_run_dir, initial_files)

        last_content = ""
        try:
            if hasattr(self, "_conversation_state") and self._conversation_state:
                last_content = self._conversation_state["messages"][-1].content
        except (KeyError, IndexError, TypeError):
            pass
        return self.log, last_content

    def _save_run_artifacts(self, run_id, run_dir, initial_files):
        """Standardized logic to save all run artifacts (trace, notebook, reports, and generated files)."""
        try:
            console.print("\n[header]💾 Saving run artifacts...[/header]")
        except NameError:
            print("\n💾 Saving run artifacts...")

        # Save trace logs (JSON)
        trace_path = os.path.join(run_dir, "trace.json")
        try:
            with open(trace_path, "w") as f:
                json.dump(self.log, f, indent=2)
            try:
                console.print("  [success]✓[/success] Saved execution trace: [bold]trace.json[/bold]")
            except NameError:
                print("  ✓ Saved execution trace: trace.json")
        except Exception as e:
            print(f"  ⚠️ Failed to save trace: {e}")

        # Save trace notebook (.ipynb)
        try:
            nb_content = self._generate_notebook()
            nb_path = os.path.join(run_dir, "trace.ipynb")
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb_content, f, indent=2)
            try:
                console.print("  [success]✓[/success] Saved trace notebook: [bold]trace.ipynb[/bold]")
            except NameError:
                print("  ✓ Saved trace notebook: trace.ipynb")
        except Exception as e:
            print(f"  ⚠️ Failed to save notebook: {e}")

        # Save report (MD and PDF)
        history_path_base = os.path.join(run_dir, "report")
        try:
            md_content = self._generate_markdown_content(include_images=True)
            with open(history_path_base + ".md", "w", encoding="utf-8") as f:
                f.write(md_content)
            try:
                console.print("  [success]✓[/success] Saved report: [bold]report.md[/bold]")
            except NameError:
                print("  ✓ Saved report: report.md")

            # Try to save PDF if possible
            self.save_conversation_history(history_path_base, include_images=True, save_pdf=True)
        except Exception as e:
            print(f"  ⚠️ Failed to save report files: {e}")

        # Move generated files
        final_files = self._get_all_files(os.getcwd())
        new_files = final_files - initial_files

        allowed_output_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
            ".svg",
            ".pdf",
            ".csv",
            ".tsv",
            ".xlsx",
            ".xls",
            ".json",
            ".jsonl",
            ".txt",
            ".md",
            ".html",
            ".parquet",
            ".npy",
            ".npz",
            ".pkl",
            ".pt",
            ".h5",
            ".hdf5",
            ".rds",
            ".loom",
            ".h5ad",
        }
        excluded_path_parts = {
            "runs",
            ".venv",
            "venv",
            "env",
            ".git",
            "__pycache__",
            ".chainlit",
            "site-packages",
            "dist-info",
            "node_modules",
        }

        def is_generated_output_file(file_path: str) -> bool:
            rel_path = os.path.relpath(file_path, os.getcwd())
            rel_parts = Path(rel_path).parts

            # Skip hidden/system/environment paths.
            if any(part.startswith(".") for part in rel_parts[:-1]):
                return False
            if any(part in excluded_path_parts for part in rel_parts):
                return False

            # Keep only likely end-user output artifact types.
            return Path(file_path).suffix.lower() in allowed_output_extensions

        new_output_files = [f for f in sorted(new_files) if is_generated_output_file(f)]

        if new_output_files:
            print(f"\n📦 New output files generated ({len(new_output_files)}):")
            for file_path in new_output_files:
                try:
                    rel_path = os.path.relpath(file_path, os.getcwd())
                    dest_path = os.path.join(run_dir, os.path.basename(file_path))

                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(dest_path)
                        dest_path = f"{base}_{int(datetime.now().timestamp())}{ext}"

                    if os.path.isfile(file_path):
                        shutil.move(file_path, dest_path)
                        print(f"  ✓ Moved to run folder: {rel_path}")
                except Exception as e:
                    print(f"  ⚠️ Failed to move {file_path}: {e}")
        else:
            print("  (No new output files generated to save)")

        try:
            console.print(Panel(f"[bold green]✅ Run {run_id} completed.[/bold green]", border_style="green"))
        except NameError:
            print(f"\n✅ Run {run_id} completed.")

    def _generate_notebook(self):
        """Generate a Jupyter Notebook structure from self.raw_log."""
        cells = []

        # Add header
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Biomni AD1 Execution Trace\n", f"Run ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}"],
            }
        )

        if not hasattr(self, "raw_log") or not self.raw_log:
            return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

        import re

        for msg in self.raw_log:
            # Handle user/system messages
            msg_type = getattr(msg, "type", "")
            msg_content = getattr(msg, "content", "")

            if msg_type in ["human", "system"]:
                cells.append(
                    {"cell_type": "markdown", "metadata": {}, "source": [f"**{msg_type.title()}**: {msg_content}"]}
                )

            # Handle AI messages (thoughts, tools, responses)
            elif msg_type == "ai":
                # 1. Look for XML tags <execute> manually (requested by trace logic)
                code_blocks = re.findall(r"<execute>(.*?)</execute>", msg_content, re.DOTALL)

                # 2. Extract thinking (text before first tag)
                thinking = msg_content
                if code_blocks:
                    first_tag_pos = msg_content.find("<execute>")
                    thinking = msg_content[:first_tag_pos].strip()

                if thinking:
                    cells.append(
                        {"cell_type": "markdown", "metadata": {}, "source": [f"**Assistant Reasoning**:\n{thinking}"]}
                    )

                for code in code_blocks:
                    cells.append(
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [code.strip()],
                        }
                    )

                # 3. Final response (text after last tag)
                after_tags = msg_content
                if code_blocks:
                    last_tag_pos = msg_content.rfind("</execute>")
                    after_tags = msg_content[last_tag_pos + 10 :].strip()

                if after_tags and not any(tag in after_tags for tag in ["<solution>", "<execute>"]):
                    cells.append(
                        {"cell_type": "markdown", "metadata": {}, "source": [f"**Assistant Output**:\n{after_tags}"]}
                    )

                # 4. Handle native tool_calls if they exist
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args")

                        if tool_name == "run_python_repl":
                            cells.append(
                                {
                                    "cell_type": "code",
                                    "execution_count": None,
                                    "metadata": {},
                                    "outputs": [],
                                    "source": [tool_args.get("command", "# No code")],
                                }
                            )
                        else:
                            cells.append(
                                {
                                    "cell_type": "markdown",
                                    "metadata": {},
                                    "source": [f"*Tool Call*: {tool_name}\nArgs: {json.dumps(tool_args)}"],
                                }
                            )

            # Handle Tool Messages (Outputs)
            elif msg_type == "tool":
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"**Observation ({getattr(msg, 'name', 'Tool')})**:\n```\n{msg_content}\n```"],
                    }
                )

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        return notebook

    def _scan_ad_catalogs_summary(self) -> str:
        """Scan BiomniAD JSON catalogs and return a compact summary string."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resource_dir = os.path.join(current_dir, "..", "know_how", "resource")

        if not os.path.isdir(resource_dir):
            return "BiomniAD catalogs: resource directory not found."

        patterns = ["BiomniAD*.json", "NIAGADS*.json", "SinaiADRD.json"]
        catalog_paths = []
        for pat in patterns:
            catalog_paths.extend(glob.glob(os.path.join(resource_dir, pat)))

        if not catalog_paths:
            return "BiomniAD catalogs: no JSON catalogs found."

        datasets = []
        for p in catalog_paths:
            try:
                with open(p) as f:
                    data = json.load(f)
                datasets.extend(data.get("datasets", []))
            except Exception:
                pass

        lines = [f"BiomniAD catalogs: {len(datasets)} datasets across {len(catalog_paths)} catalog file(s)."]
        lines.append("To browse them, load JSON files from: " + resource_dir)
        for d in datasets[:20]:
            title = d.get("title", d.get("id", "?"))
            n_files = len(d.get("files", []))
            lines.append(f"  - {title} ({n_files} file URI(s))")
        if len(datasets) > 20:
            lines.append(f"  ... and {len(datasets) - 20} more datasets")

        # Check for local presence of files
        enriched_lines = []
        ad_data_lake = os.path.join(getattr(self, "data_lake_dir", ""), "biomniAD")

        for line in lines:
            if line.startswith("  - "):
                # Try to extract the title/id from the line and dataset object to check local files
                # This matches the title in the loop below
                pass
            enriched_lines.append(line)

        # Better loop for checking local availability
        final_lines = [lines[0], lines[1]]
        for d in datasets[:25]:
            title = d.get("title", d.get("id", "?"))
            ds_id = d.get("id", "unknown")
            files = d.get("files", [])

            local_count = 0
            if ad_data_lake and os.path.isdir(os.path.join(ad_data_lake, ds_id)):
                ds_dir = os.path.join(ad_data_lake, ds_id)
                for f in files:
                    fname = f.get("name", os.path.basename(f.get("uri", "")))
                    if fname and os.path.exists(os.path.join(ds_dir, fname)):
                        local_count += 1

            status = f" ({local_count}/{len(files)} LOCAL)" if local_count > 0 else f" ({len(files)} URIs)"
            final_lines.append(f"  - {title}{status}")

        if len(datasets) > 25:
            final_lines.append(f"  ... and {len(datasets) - 25} more datasets")

        return "\n".join(final_lines)

    def _inject_ad_context(self):
        """Inject BiomniAD data sourcing instructions into the system prompt."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            know_how_path = os.path.join(current_dir, "..", "know_how", "biomniAD_data_sourcing.md")

            if os.path.exists(know_how_path):
                with open(know_how_path) as f:
                    ad_sourcing_content = f.read()

                data_root_dir = getattr(self, "data_root_dir", "not set")

                ad_instruction = f"""

AD/DEMENTIA TOOL PRIORITY — ALWAYS FOLLOW THIS ORDER:
1. **Local data first**: Scan the built-in data lake ({getattr(self, "data_lake_dir", "not set")}) and user data directory ({data_root_dir}) for any locally available AD datasets.
   Use os.listdir() on both locations — the data lake has curated datasets; the user directory may contain additional data.
2. **BiomniAD catalogs second**: Load JSON catalogs from biomni/know_how/resource/ to find datasets with download URIs.
3. **Web & literature search third**: Use advanced_web_search(), search_pubmed(), search_biorxiv() to supplement.
4. **Code generation last**: Write custom Python/R code only when the above cannot answer directly.
Do NOT simulate or fabricate data at any step.

ALZHEIMER'S & DEMENTIA DATA SOURCING PROTOCOL
========================================================
{ad_sourcing_content}
========================================================
"""
                self.system_prompt += ad_instruction
                self._enforce_local_data_priority()
            else:
                print(f"Warning: Could not find AD data sourcing guide at {know_how_path}")

        except Exception as e:
            print(f"Warning: Failed to inject AD context: {e}")

    def launch_ui(
        self,
        thread_id: int = 42,
        share: bool = False,
        server_name: str = "0.0.0.0",
        require_verification: bool = False,
    ) -> None:
        """Launch the Biomni AD1 Web UI."""
        try:
            import gradio as gr
            from gradio import ChatMessage
        except ImportError as exc:
            raise ImportError("Gradio is not installed. Please install it with: pip install gradio") from exc

        import re
        from time import time

        supported_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf")

        self.main_history_copy = []
        available_access_codes = ["Biomni2025"]

        def verify_access_code(code: str):
            if code in available_access_codes:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(
                    value="Incorrect access code. Please check your access code.",
                    visible=True,
                ),
            )

        def get_all_files(directory):
            """Recursively get all files in a directory."""
            file_list = []
            for root, _dirs, files in os.walk(directory):
                for file in files:
                    # Ignore hidden files and directories
                    if file.startswith(".") or "/." in root:
                        continue
                    file_list.append(os.path.join(root, file))
            return set(file_list)

        def get_runs_list():
            """Get list of recent runs with prompt history."""
            runs_dir = os.path.join(os.getcwd(), "runs")
            if not os.path.exists(runs_dir):
                return "No runs found yet."

            # Sort runs by name (timestamped)
            runs = sorted([d for d in os.listdir(runs_dir) if d.startswith("run_")], reverse=True)

            md_list = "### 🕒 Recent Runs\n"
            for i, r in enumerate(runs[:15]):  # Show up to 15
                run_num = len(runs) - i
                prompt_snippet = ""

                # Try to get prompt from trace.json
                trace_json_path = os.path.join(runs_dir, r, "trace.json")
                if os.path.exists(trace_json_path):
                    try:
                        with open(trace_json_path) as f:
                            trace_data = json.load(f)
                            if trace_data and isinstance(trace_data, list):
                                # First entry is usually the user prompt
                                first_msg = trace_data[0]
                                if "Human Message" in first_msg:
                                    # Extract text after headers
                                    prompt_snippet = first_msg.split("\n\n")[-1][:60].strip() + "..."
                    except Exception:
                        pass

                if prompt_snippet:
                    md_list += f"> **Run #{run_num}**  \n> {prompt_snippet}  \n> [ {r} ]\n\n"
                else:
                    md_list += f"**Run #{run_num}** (`{r}`)\n\n"
            return md_list

        def get_local_data_summary():
            """Get a brief markdown summary of local data available to AD1."""
            try:
                local_items = self._get_data_lake_items() if hasattr(self, "_get_data_lake_items") else []
            except Exception:
                local_items = []

            lines = ["### 💾 Local Data", f"**{len(local_items)} data lake file(s)**"]
            preview_limit = 20
            if local_items:
                for item in local_items[:preview_limit]:
                    lines.append(f"- `{item}`")
                if len(local_items) > preview_limit:
                    lines.append(f"- ... and {len(local_items) - preview_limit} more")
            else:
                lines.append("- No data lake files detected yet")

            if hasattr(self, "data_lake_dir"):
                lines.append(f"\nData lake: `{self.data_lake_dir}`")

            # Show data root directory contents
            data_root = getattr(self, "data_root_dir", None)
            if data_root and os.path.isdir(data_root):
                lines.append("\n### 📂 Data Root")
                lines.append(f"Path: `{data_root}`")
                try:
                    entries = sorted([name for name in os.listdir(data_root) if not name.startswith(".")])
                except OSError:
                    entries = []

                if entries:
                    for name in entries[:15]:
                        full = os.path.join(data_root, name)
                        suffix = "/" if os.path.isdir(full) else ""
                        lines.append(f"- `{name}{suffix}`")
                    if len(entries) > 15:
                        lines.append(f"- ... and {len(entries) - 15} more")
                else:
                    lines.append("- (empty)")

            user_data_path = os.getenv("BIOMNI_USER_DATA_PATH", "").strip()
            if user_data_path:
                lines.append("\n### 👤 User Data Folder")
                lines.append(f"Path: `{user_data_path}`")
                if os.path.isdir(user_data_path):
                    try:
                        entries = sorted([name for name in os.listdir(user_data_path) if not name.startswith(".")])
                    except OSError:
                        entries = []

                    if entries:
                        preview_user_limit = 10
                        for name in entries[:preview_user_limit]:
                            lines.append(f"- `{name}`")
                        if len(entries) > preview_user_limit:
                            lines.append(f"- ... and {len(entries) - preview_user_limit} more")
                    else:
                        lines.append("- (empty)")
                else:
                    lines.append("- (path not found)")
            return "\n".join(lines)

        def refresh_sidebar():
            return get_runs_list(), get_local_data_summary()

        def generate_response(prompt_input, inner_history=None, main_history=None):
            if main_history is None:
                main_history = []
            if inner_history is None:
                inner_history = []

            text_input = prompt_input.get("text", "")
            files = prompt_input.get("files", [])

            # Capture initial file state
            initial_files = get_all_files(os.getcwd())

            self.main_history_copy += [{"role": "user", "content": text_input}]
            main_history.append(
                ChatMessage(
                    role="user",
                    content=text_input if text_input else "[Uploaded file]",
                    metadata={"title": "👤 User"},
                )
            )

            main_history.append(
                ChatMessage(
                    role="assistant",
                    content="Executor is working on it 👉",
                    metadata={"title": "🛠️ Planning"},
                )
            )
            # AD1 Specific: Inject AD context if keywords are present
            is_ad_task = any(keyword.lower() in text_input.lower() for keyword in self.ad_keywords)
            if is_ad_task:
                print("\n🧠 AD/Dementia task detected. Injecting specialized data sourcing protocols...")
                self._inject_ad_context()
                inner_history.append(
                    ChatMessage(
                        role="assistant",
                        content="🧠 AD/Dementia task detected. Injecting specialized data sourcing protocols...",
                        metadata={"title": "🧠 AD Context"},
                    )
                )
                yield (
                    inner_history,
                    main_history,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )  # Update outputs including runs_list

            for file_info in files:
                file_path = file_info
                text_input += f"\n\n User uploaded this file: {file_path}\n Please use it if needed."

            agent_messages = []
            for msg in self.main_history_copy:
                if msg["role"] == "user":
                    agent_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant" and msg["content"] not in ["Executor is working on it 👉"]:
                    from langchain_core.messages import AIMessage

                    agent_messages.append(AIMessage(content=msg["content"]))

            agent_messages.append(HumanMessage(content=text_input))

            inputs = {"messages": agent_messages, "next_step": None}
            config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}

            t = time()
            solution_found = False

            # Check for tool retriever if it exists (A1 feature)
            if getattr(self, "use_tool_retriever", False):
                inner_history.append(
                    ChatMessage(
                        role="assistant",
                        content="Retrieving relevant tools, data lake items, and libraries...",
                        metadata={"title": "🔍 Tool retrieval"},
                    )
                )
                yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                try:
                    selected_resources_names = self._prepare_resources_for_retrieval(text_input)
                    if selected_resources_names:
                        self.update_system_prompt_with_selected_resources(selected_resources_names)
                except Exception as exc:
                    print(f"Warning: Tool retrieval failed: {exc}")
                    inner_history.append(
                        ChatMessage(
                            role="assistant",
                            content="Tool retrieval unavailable, proceeding with all tools...",
                            metadata={"title": "⚠️ Tool retrieval"},
                        )
                    )
                    yield inner_history, main_history, gr.update(), gr.update(), gr.update()

            code_execution_messages = []

            for s in self.app.stream(inputs, stream_mode="values", config=config):
                t_step = time() - t
                message = s["messages"][-1]

                if message.content == text_input:
                    t = time()
                    continue

                if isinstance(message.content, str):
                    tag_positions = [
                        pos
                        for tag in ["<execute>", "<solution>", "<observation>"]
                        if (pos := message.content.find(tag)) != -1
                    ]

                    if tag_positions:
                        first_tag_pos = min(tag_positions)
                        thinking = message.content[:first_tag_pos].strip()
                        if thinking:
                            inner_history.append(
                                ChatMessage(
                                    role="assistant",
                                    content=thinking,
                                    metadata={"title": "🤔 Reasoning"},
                                )
                            )
                            yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                    solution_match = re.search(r"<solution>(.*?)</solution>", message.content, re.DOTALL)
                    if solution_match and not solution_found:
                        solution_found = True
                        solution = solution_match.group(1).strip()
                        main_history.append(
                            ChatMessage(
                                role="assistant",
                                content=solution,
                                metadata={"title": "✅ Answer", "log": "Final answer"},
                            )
                        )
                        self.main_history_copy += [{"role": "assistant", "content": solution}]
                        yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                    execute_match = re.search(r"<execute>(.*?)</execute>", message.content, re.DOTALL)
                    if execute_match:
                        code = execute_match.group(1).strip()
                        language = "python"
                        if code.strip().startswith("#!R"):
                            language = "r"
                            code = re.sub(r"^#!R", "", code, count=1).strip()
                        elif code.strip().startswith("#!BASH") or code.strip().startswith("#!CLI"):
                            language = "bash"
                            code = re.sub(r"^#!BASH|^#!CLI", "", code, count=1).strip()

                        code_msg = ChatMessage(
                            role="assistant",
                            content=f"##### Code: \n```{language}\n{code}\n```",
                            metadata={
                                "title": "🛠️ Executing code...",
                                "status": "pending",
                                "start_time": t,
                            },
                        )
                        inner_history.append(code_msg)
                        code_execution_messages.append(code_msg)
                        yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                    observation_match = re.search(r"<observation>(.*?)</observation>", message.content, re.DOTALL)
                    if observation_match:
                        observation = observation_match.group(1).strip()

                        if code_execution_messages:
                            code_msg = code_execution_messages[-1]
                            code_msg.metadata.update(
                                {
                                    "status": "done",
                                    "duration": t_step,
                                    "log": f"Code execution completed in {t_step:.2f}s",
                                }
                            )

                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content=f"##### Observation: \n```\n{observation}\n```",
                                metadata={
                                    "status": "done",
                                    "duration": t_step,
                                    "log": "Observation from code execution",
                                    "collapsed": True,
                                    "collapsible": True,
                                },
                            )
                        )
                        yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                        if isinstance(observation, str) and any(ext in observation for ext in supported_extensions):
                            matches = re.findall(
                                r"(\S+?(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp|\.pdf))",
                                observation,
                            )
                            valid_matches = []
                            for match in matches:
                                if not (
                                    match.startswith("Warning:") or match.startswith("Error:") or match.startswith("'")
                                ):
                                    if not match.startswith("."):
                                        valid_matches.append(match)

                            if valid_matches:
                                inner_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content="",
                                        metadata={"title": "📁 Files"},
                                    )
                                )

                                for file_path in valid_matches:
                                    file_path = file_path.strip("\"'").strip()
                                    abs_path = None
                                    if os.path.isabs(file_path) and os.path.exists(file_path):
                                        abs_path = file_path
                                    elif os.path.exists(os.path.join(os.getcwd(), file_path)):
                                        abs_path = os.path.join(os.getcwd(), file_path)
                                    elif (
                                        hasattr(self, "path")
                                        and self.path
                                        and os.path.exists(os.path.join(self.path, file_path))
                                    ):
                                        abs_path = os.path.join(self.path, file_path)

                                    if abs_path:
                                        if file_path.lower().endswith(".pdf"):
                                            inner_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content=f"Found PDF at: {abs_path}",
                                                    metadata={"title": "📄 PDF File"},
                                                )
                                            )
                                        else:
                                            inner_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content=gr.Image(abs_path),
                                                    metadata={"title": "🖼️ Image Preview"},
                                                )
                                            )
                                yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                t = time()

            final_message = s["messages"][-1].content if s.get("messages") else ""
            if not solution_found:
                solution_match = re.search(r"<solution>(.*?)</solution>", final_message, re.DOTALL)
                if solution_match:
                    solution = solution_match.group(1).strip()
                    main_history.append(
                        ChatMessage(
                            role="assistant",
                            content=solution,
                            metadata={"title": "✅ Solution"},
                        )
                    )
                    self.main_history_copy += [{"role": "assistant", "content": solution}]
                else:
                    cleaned_content = re.sub(r"<execute>.*?</execute>", "", final_message, flags=re.DOTALL)
                    cleaned_content = re.sub(r"<observation>.*?</observation>", "", cleaned_content, flags=re.DOTALL)
                    cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)

                    summary = cleaned_content.strip() or ("Task completed. Please check the execution log for details.")
                    main_history.append(
                        ChatMessage(
                            role="assistant",
                            content=summary,
                            metadata={"title": "📝 Summary"},
                        )
                    )
                    self.main_history_copy += [{"role": "assistant", "content": summary}]

            inner_history.append(
                ChatMessage(
                    role="assistant",
                    content="👈 Returning the result to the main interface...",
                    metadata={"title": "🔄 Complete"},
                )
            )

            # Sync raw_log for notebook generation
            if s and "messages" in s:
                self.raw_log = list(s["messages"])

            # Setup run directory
            run_id = self._build_run_id(prompt_input)
            runs_root = os.path.abspath(os.path.join(os.getcwd(), "runs"))
            os.makedirs(runs_root, exist_ok=True)  # Ensure runs_root exists
            current_run_dir = os.path.join(runs_root, run_id)
            os.makedirs(current_run_dir, exist_ok=True)

            # Centralized artifact saving
            self._save_run_artifacts(run_id, current_run_dir, initial_files)

            # Update runs list in sidebar
            updated_runs_list = get_runs_list()

            # Count total runs for the success message
            total_runs = len([d for d in os.listdir(runs_root) if d.startswith("run_")])

            status_md = f"### ✅ Run #{total_runs} Completed\nAll artifacts, including logs and generated data, have been moved to:\n`{current_run_dir}`"

            yield (
                inner_history,
                main_history,
                gr.update(value=status_md, visible=True),
                gr.update(value=updated_runs_list),
                gr.update(visible=False),
            )

        def like(data: Any = None) -> None:
            """Handle like/dislike events from the chatbot."""
            if data is not None:
                print("User liked the response")

        # Custom CSS - Simple & Professional
        custom_css = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        body, .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #ffffff !important;
        }

        /* Sidebar */
        #sidebar {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
            padding: 20px;
            min-width: 240px !important;
        }

        #sidebar .prose {
            word-wrap: break-word !important;
            white-space: normal !important;
        }

        #sidebar .prose blockquote {
            border-left: 3px solid #3b82f6;
            margin: 8px 0;
            padding: 10px 12px;
            background: #ffffff;
            border-radius: 0 6px 6px 0;
            font-size: 13px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }

        #sidebar .prose strong {
            color: #1e40af;
        }

        /* Chat Area */
        #chat-area {
            background: #ffffff;
            padding: 24px;
        }

        /* Trace Area */
        #trace-area {
            background: #fafafa;
            border-left: 1px solid #e2e8f0;
            padding: 20px;
        }

        /* Messages */
        .message-row.user-row .message {
            background: #2563eb !important;
            color: white !important;
            border-radius: 16px 16px 4px 16px !important;
        }

        .message-row.bot-row .message {
            background: #f1f5f9 !important;
            border-radius: 16px 16px 16px 4px !important;
        }

        /* Buttons */
        button.primary {
            background: #2563eb !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }

        button.secondary {
            background: #f1f5f9 !important;
            color: #1e40af !important;
            border: 1px solid #e2e8f0 !important;
        }

        .prose {
            font-size: 14px !important;
            line-height: 1.5 !important;
            color: #1e293b !important;
        }
        """

        with gr.Blocks(title="Biomni AD1", theme=gr.themes.Soft(), css=custom_css) as demo:
            # AD1 Logo
            logo_path = Path(__file__).resolve().parents[2] / "figs" / "Biomni-AD_Logo_v2.png"
            if not logo_path.exists():
                logo_path = Path(__file__).resolve().parents[2] / "figs" / "biomni_logo.png"

            verification_container = gr.Group(visible=require_verification)
            main_interface_container = gr.Group(visible=not require_verification)

            with verification_container:
                if logo_path.exists():
                    gr.Image(logo_path, show_label=False, height=80)
                gr.Markdown("## Access Verification")
                access_code_input = gr.Textbox(label="Access Code", type="password")
                access_error_msg = gr.Markdown(visible=False)
                verify_btn = gr.Button("Verify", variant="primary")
                verify_btn.click(
                    fn=verify_access_code,
                    inputs=[access_code_input],
                    outputs=[verification_container, main_interface_container, access_error_msg],
                )

            with main_interface_container:
                # Top Header (Logo + Title)
                with gr.Row(elem_classes="header-row"):
                    with gr.Column(scale=1):
                        if logo_path.exists():
                            gr.Image(logo_path, show_label=False, height=60, container=False)
                        else:
                            gr.Markdown("# Biomni AD1")

                with gr.Row(elem_id="main-row"):
                    # --- Left Sidebar: Explorer ---
                    with gr.Column(scale=1, elem_id="sidebar"):
                        gr.Markdown("## 📂 Explorer")

                        runs_list = gr.Markdown(value=get_runs_list())
                        local_data_summary = gr.Markdown(value=get_local_data_summary())
                        refresh_runs_btn = gr.Button("Refresh", size="sm", variant="secondary")
                        refresh_runs_btn.click(fn=refresh_sidebar, outputs=[runs_list, local_data_summary])

                    # --- Center: Agent Chat ---
                    with gr.Column(scale=3, elem_id="chat-area"):
                        main_chatbot = gr.Chatbot(
                            label="Conversation",
                            type="messages",
                            height=650,
                            show_copy_button=True,
                            show_share_button=True,
                            autoscroll=True,
                            avatar_images=(None, None),  # Can add avatars here
                            elem_id="main-chatbot",
                        )

                        with gr.Row():
                            prompt_input = gr.MultimodalTextbox(
                                interactive=True,
                                file_count="multiple",
                                placeholder="Ask a research question or upload data...",
                                show_label=False,
                                scale=8,
                            )
                            # submit_btn = gr.Button("Send", variant="primary", scale=1) # MultimodalTextbox has embed submit

                    # --- Right: Execution Trace ---
                    with gr.Column(scale=2, elem_id="trace-area"):
                        gr.Markdown("## ⚡ Execution Trace")
                        innerloop_chatbot = gr.Chatbot(
                            label="Thought Process",
                            type="messages",
                            height=500,
                            show_copy_button=True,
                            elem_id="inner-chatbot",
                        )

                        gr.Markdown("### 📦 Run Artifacts")
                        run_status = gr.Markdown("Waiting for execution...")

                        # Hidden placeholders
                        pdf_output = gr.File(visible=False)
                        gr.File(visible=False)

                # Wiring
                prompt_input.submit(
                    generate_response,
                    [prompt_input, innerloop_chatbot, main_chatbot],
                    [innerloop_chatbot, main_chatbot, run_status, runs_list, pdf_output],
                ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])

                main_chatbot.like(like)

        print(f"Launching Biomni AD1 Gradio demo on {server_name}:7860")
        demo.launch(share=share, server_name=server_name)
