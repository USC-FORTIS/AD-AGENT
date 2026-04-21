#!/usr/bin/env python3
import os
import sys
import time
import json
import builtins
import logging
from typing import ClassVar
import torch
import numpy as np
import pandas as pd
import glob

# ─── Bootstrap API key & logging ───────────────────────────────────────────────
from config.config import Config
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# ─── Instrumentation helpers ──────────────────────────────────────────────────
def _unpack_usage(usage):
    if usage is None:
        return 0, 0, 0
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        ct = usage.get("completion_tokens", usage.get("output_tokens", 0))
        tt = usage.get("total_tokens", pt + ct)
    else:
        pt = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0)
        ct = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0)
        tt = getattr(usage, "total_tokens", pt + ct)
    return pt, ct, tt

# ─── Patch LangChain's ChatOpenAI ────────────────────────────────────────────
import langchain_openai
BaseChat = langchain_openai.ChatOpenAI

class InstrumentedChatOpenAI(BaseChat):
    prompt_tokens:     ClassVar[int] = 0
    completion_tokens: ClassVar[int] = 0
    total_tokens:      ClassVar[int] = 0

    def _call(self, messages, **kwargs):
        resp = super()._call(messages, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, _ = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
        return resp

    def __call__(self, *args, **kwargs):
        resp = super().__call__(*args, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, _ = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
        return resp

    def invoke(self, prompt, **kwargs):
        resp = super().invoke(prompt, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, _ = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
        return resp

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        usage = result.llm_output.get("usage") or result.llm_output.get("token_usage")
        if usage:
            pt, ct, _ = _unpack_usage(usage)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
        return result

langchain_openai.ChatOpenAI = InstrumentedChatOpenAI

# ─── Patch openai.OpenAI ─────────────────────────────────────────────────────
import openai
BaseOpenAI = openai.OpenAI

class InstrumentedOpenAI(BaseOpenAI):
    prompt_tokens:     ClassVar[int] = 0
    completion_tokens: ClassVar[int] = 0
    total_tokens:      ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        orig = self.chat.completions.create
        def wrapped(*a, **k):
            resp = orig(*a, **k)
            if (u := getattr(resp, "usage", None)):
                pt, ct, _ = _unpack_usage(u)
                InstrumentedOpenAI.prompt_tokens     += pt
                InstrumentedOpenAI.completion_tokens += ct
            return resp
        self.chat.completions.create = wrapped

        if hasattr(self, "responses") and hasattr(self.responses, "create"):
            orig2 = self.responses.create
            def wrapped2(*a, **k):
                resp = orig2(*a, **k)
                if (u := getattr(resp, "usage", None)):
                    pt, ct, _ = _unpack_usage(u)
                    InstrumentedOpenAI.prompt_tokens     += pt
                    InstrumentedOpenAI.completion_tokens += ct
                return resp
            self.responses.create = wrapped2

openai.OpenAI = InstrumentedOpenAI
_global_client = openai.OpenAI()
openai.chat.completions.create = _global_client.chat.completions.create

def reset_counters():
    for C in (InstrumentedChatOpenAI, InstrumentedOpenAI):
        C.prompt_tokens     = 0
        C.completion_tokens = 0
        C.total_tokens      = 0

# ─── Import your agents ──────────────────────────────────────────────────────
from agents.agent_processor import AgentProcessor
from agents.agent_selector import AgentSelector
from agents.agent_info_miner import AgentInfoMiner
from agents.agent_code_generator import AgentCodeGenerator
from agents.agent_reviewer import AgentReviewer
from utils.script_paths import benchmark_script_path

# ─── Benchmark configuration ──────────────────────────────────────────────────
DATA_DIR = "./data/pygod_data"
ALGOS = ["GANN"]

def run_one(algo: str, train_file: str):
    workflow_start = time.perf_counter()
    
    row = {
        "algorithm": algo,
        "dataset": os.path.basename(train_file),
        "success": True,
        "error": "",
        "time_sec": 0,
        "total_time_sec": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "processor_input_tokens": 0,
        "processor_output_tokens": 0,
        "selector_input_tokens": 0,
        "selector_output_tokens": 0,
        "infominer_input_tokens": 0,
        "infominer_output_tokens": 0,
        "codegen_input_tokens": 0,
        "codegen_output_tokens": 0,
        "reviewer_input_tokens": 0,
        "reviewer_output_tokens": 0,
        "revision_count": 0
    }

    # ─ Processor ──────────────────────────────────────────────────────────────
    reset_counters()
    prompt = f"Run {algo} on {train_file}"
    orig_input = builtins.input
    def fake_input(prompt_text=""):
        print(prompt_text, end="")
        return prompt
    builtins.input = fake_input

    proc_start = time.perf_counter()
    proc = AgentProcessor(model="gpt-4", temperature=0)
    try:
        proc.run_chatbot()
        row["success"] = True
        row["error"]   = ""
    except Exception as e:
        row["success"] = False
        row["error"]   = str(e)
    finally:
        builtins.input = orig_input
    proc_end = time.perf_counter()

    row["time_sec"] = proc_end - proc_start
    row["processor_input_tokens"] = InstrumentedOpenAI.prompt_tokens + InstrumentedChatOpenAI.prompt_tokens
    row["processor_output_tokens"] = InstrumentedOpenAI.completion_tokens + InstrumentedChatOpenAI.completion_tokens
    row["total_input_tokens"] += row["processor_input_tokens"]
    row["total_output_tokens"] += row["processor_output_tokens"]

    cfg = proc.experiment_config

    # ─ Selector ────────────────────────────────────────────────────────────────
    reset_counters()
    sel = AgentSelector(cfg)
    row["selector_input_tokens"] = InstrumentedChatOpenAI.prompt_tokens + InstrumentedOpenAI.prompt_tokens
    row["selector_output_tokens"] = InstrumentedChatOpenAI.completion_tokens + InstrumentedOpenAI.completion_tokens
    row["total_input_tokens"] += row["selector_input_tokens"]
    row["total_output_tokens"] += row["selector_output_tokens"]

    # ─ InfoMiner ───────────────────────────────────────────────────────────────
    reset_counters()
    inf = AgentInfoMiner()
    doc = inf.query_docs(algo, sel.vectorstore, sel.package_name)
    row["infominer_input_tokens"] = InstrumentedOpenAI.prompt_tokens
    row["infominer_output_tokens"] = InstrumentedOpenAI.completion_tokens
    row["total_input_tokens"] += row["infominer_input_tokens"]
    row["total_output_tokens"] += row["infominer_output_tokens"]

    # ─ Code Generator ─────────────────────────────────────────────────────────
    reset_counters()
    cg = AgentCodeGenerator()
    code = cg.generate_code(
        algorithm        = algo,
        data_path_train  = train_file,
        data_path_test   = "",  # No test file
        algorithm_doc    = doc,
        input_parameters = sel.parameters,
        package_name     = sel.package_name
    )
    row["codegen_input_tokens"] = InstrumentedChatOpenAI.prompt_tokens
    row["codegen_output_tokens"] = InstrumentedChatOpenAI.completion_tokens
    row["total_input_tokens"] += row["codegen_input_tokens"]
    row["total_output_tokens"] += row["codegen_output_tokens"]

    # Save the generated code to a file
    dataset_name = os.path.splitext(os.path.basename(train_file))[0]
    path = benchmark_script_path(algo, dataset_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\n=== [Code Generator] Saved code to {path} ===")

    # ─ Reviewer and Code Revision Loop ────────────────────────────────────────
    max_revisions = 10  # Maximum number of revision attempts
    revision_count = 0
    error = None
    
    while revision_count < max_revisions:
        reset_counters()
        rev = AgentReviewer()
        error = rev.test_code(code, algo, sel.package_name)
        row["reviewer_input_tokens"] = InstrumentedChatOpenAI.prompt_tokens
        row["reviewer_output_tokens"] = InstrumentedChatOpenAI.completion_tokens
        row["total_input_tokens"] += row["reviewer_input_tokens"]
        row["total_output_tokens"] += row["reviewer_output_tokens"]
        
        if not error:  # If no error, break the loop
            break
            
        # If there's an error, revise the code
        revision_count += 1
        row["revision_count"] = revision_count
        print(f"\n=== [Code Generator] Revising code (attempt {revision_count}) ===")
        
        # Revise the code
        reset_counters()
        revised_code = cg.revise_code(code, error)
        row["codegen_input_tokens"] += InstrumentedChatOpenAI.prompt_tokens
        row["codegen_output_tokens"] += InstrumentedChatOpenAI.completion_tokens
        row["total_input_tokens"] += InstrumentedChatOpenAI.prompt_tokens
        row["total_output_tokens"] += InstrumentedChatOpenAI.completion_tokens
        code = revised_code
        
        # Save the revised code
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"\n=== [Code Generator] Saved revised code to {path} ===")
    
    if error:
        row["success"] = False
        row["error"] = error

    workflow_end = time.perf_counter()
    row["total_time_sec"] = workflow_end - workflow_start

    return row

def main():
    all_results = []
    train_files = glob.glob(os.path.join(DATA_DIR, "*.pt"))  # PyGOD uses .pt files
    
    for algo in ALGOS:
        print(f"\n=== Running {algo} ===")
        for train_file in train_files:
            print(f"\n=== Processing {os.path.basename(train_file)} ===")
            result = run_one(algo, train_file)
            all_results.append(result)
            
            # Print detailed token counts for this run
            print("\n=== Token Usage Summary ===")
            print(f"Processor: {result['processor_input_tokens']} in, {result['processor_output_tokens']} out")
            print(f"Selector: {result['selector_input_tokens']} in, {result['selector_output_tokens']} out")
            print(f"InfoMiner: {result['infominer_input_tokens']} in, {result['infominer_output_tokens']} out")
            print(f"CodeGenerator: {result['codegen_input_tokens']} in, {result['codegen_output_tokens']} out")
            print(f"Reviewer: {result['reviewer_input_tokens']} in, {result['reviewer_output_tokens']} out")
            print(f"Total: {result['total_input_tokens']} in, {result['total_output_tokens']} out")
            print("=" * 50)

    df = pd.DataFrame(all_results)
    
    # Create a 2D table with algorithms as rows and datasets as columns
    metrics = ['total_time_sec', 'total_input_tokens', 'total_output_tokens', 'revision_count']
    
    for metric in metrics:
        pivot_df = df.pivot(index='algorithm', columns='dataset', values=metric)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_fn = f"benchmark_pygod_{metric}_{ts}.csv"
        pivot_df.to_csv(out_fn)
        print(f"\nSaved {metric} results to {out_fn}")

    # Save full results
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_fn = f"benchmark_pygod_full_{ts}.csv"
    df.to_csv(out_fn, index=False)
    print(f"\nSaved full results to {out_fn}")

if __name__ == "__main__":
    main() 
