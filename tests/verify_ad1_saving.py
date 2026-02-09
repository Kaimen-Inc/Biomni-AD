
import os
import shutil
import json
from biomni.agent.ad1 import AD1

def test_ad1_tracking():
    print("Testing AD1 tracking...")
    
    # Clean up previous runs if any (optional, but good for clean test)
    # runs_dir = os.path.join(os.getcwd(), "biomni_data", "runs")
    # if os.path.exists(runs_dir):
    #     shutil.rmtree(runs_dir)

    # Instantiate AD1
    # We might need to mock some things if real execution is costly or requires API keys
    # But let's try a real run with a simple query if possible, or mock LLM if needed.
    # Assuming A1/AD1 needs an LLM. If no keys are present, it might fail.
    # Let's check environment.
    
    try:
        # Pass expected_data_lake_files=[] to skip massive download
        agent = AD1(expected_data_lake_files=[])
    except Exception as e:
        print(f"Failed to instantiate AD1: {e}")
        return

    # Run a simple query
    query = "Calculate 1 + 1. Please respond with just the number."
    print(f"Running query: {query}")
    
      # Run the agent with a prompt that generates a file + needs code execution
    prompt = "Calculate 123 + 456 using python code, and then create a file named 'test_artifact.txt' with the result inside it."
    result = agent.go(prompt)
    print("Agent returned result.")
    
    # Verification
    # User requested runs be in the CWD/runs
    runs_dir = os.path.join(os.getcwd(), "runs")
    if not os.path.exists(runs_dir):
        print(f"FAIL: runs directory not found at {runs_dir}")
        return

    # Find the latest run
    runs = sorted(os.listdir(runs_dir))
    if not runs:
        print("FAIL: No run directory created.")
        return
        
    latest_run = runs[-1]
    run_path = os.path.join(runs_dir, latest_run)
    print(f"Checking run directory: {run_path}")
    
    # 1. Check Logical Artifacts
    trace_path = os.path.join(run_path, "trace.json")
    if os.path.exists(trace_path):
        print("PASS: trace.json found.")
        # Verify trace content has code
        with open(trace_path, "r") as f:
            trace_content = f.read()
            if "tool_calls" in trace_content or "Tool Call" in trace_content or "123 + 456" in trace_content:
                print("PASS: trace.json contains code/tool info.")
            else:
                 print("WARNING: trace.json might be missing code details. Check content.")
    else:
        print("FAIL: trace.json missing.")

    if os.path.exists(os.path.join(run_path, "report.md")):
        print("PASS: report.md found.")
    else:
        print("FAIL: report.md missing.")

    # 1b. Check Notebook Artifact
    nb_path = os.path.join(run_path, "trace.ipynb")
    if os.path.exists(nb_path):
        print("PASS: trace.ipynb found.")
        with open(nb_path, "r") as f:
            nb_content = json.load(f)
            if "cells" in nb_content and len(nb_content["cells"]) > 0:
                print(f"PASS: trace.ipynb is valid and has {len(nb_content['cells'])} cells.")
                # Look for code cell
                has_code = any(cell["cell_type"] == "code" for cell in nb_content["cells"])
                if has_code:
                     print("PASS: trace.ipynb contains code cells.")
                else:
                     print("WARNING: trace.ipynb has no code cells (Expected 123+456 calculation).")
            else:
                print("FAIL: trace.ipynb is empty or invalid.")
    else:
        print("FAIL: trace.ipynb missing.")
        
    # 2. Check Generated File Moving
    artifact_path_run = os.path.join(run_path, "test_artifact.txt")
    artifact_path_cwd = "test_artifact.txt"
    
    if os.path.exists(artifact_path_run):
        print(f"PASS: Generated file 'test_artifact.txt' found in run directory.")
    else:
        print(f"FAIL: Generated file 'test_artifact.txt' NOT found in run directory.")
        
    if not os.path.exists(artifact_path_cwd):
        print(f"PASS: Generated file 'test_artifact.txt' correctly removed from CWD.")
    else:
        print(f"FAIL: Generated file 'test_artifact.txt' still exists in CWD (Should have been moved).")

if __name__ == "__main__":
    test_ad1_tracking()
