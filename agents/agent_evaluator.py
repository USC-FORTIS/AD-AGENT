import os, re, subprocess, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from entity.code_quality import CodeQuality

class AgentEvaluator:
    """
    Executes the final code with real data and parses AUROC/AUPRC.
    (Logic ported from the old Reviewer.execute_code)
    """

    # ---------- public ----------
    def execute_code(self, code: str, algorithm_name: str) -> CodeQuality:
        # Create folder for generated scripts if it doesn't exist
        folder = "./generated_scripts"
        os.makedirs(folder, exist_ok=True)

        # Save the provided code to a Python file
        path = os.path.join(folder, f"{algorithm_name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        # Execute the script using subprocess and capture output
        res = subprocess.run(["python", path], capture_output=True, text=True)
        print("\n=== Real-Data Execution Output ===\n", res.stdout, res.stderr)

        # If execution failed, return error result
        if res.returncode != 0:
            return CodeQuality(
                code=code, algorithm=algorithm_name, parameters={}, std_output="",
                error_message=self._subprocess_output_as_error(res.stdout, res.stderr),
                auroc=-1, auprc=-1, error_points=[], review_count=0
            )

        nested_error = self._detect_nested_failure(res.stdout, res.stderr)
        if nested_error:
            return CodeQuality(
                code=code, algorithm=algorithm_name, parameters={}, std_output=res.stdout,
                error_message=self._subprocess_output_as_error(res.stdout, res.stderr),
                auroc=-1, auprc=-1, error_points=[], review_count=0
            )

        # Parse metrics from the script output
        auroc  = self._find_float(r"AUROC:\s*([\d.]+)", res.stdout)
        auprc  = self._find_float(r"AUPRC:\s*([\d.]+)", res.stdout)
        errors = self._parse_errors(res.stdout)

        # Return evaluation result
        return CodeQuality(
            code=code, algorithm=algorithm_name, parameters={}, std_output=res.stdout,
            error_message="", auroc=auroc, auprc=auprc,
            error_points=errors, review_count=0
        )

    # ---------- helpers ----------
    @staticmethod
    def _find_float(pattern: str, text: str, default: float = -1.0) -> float:
        # Find a float value in the text using regex
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _parse_errors(text: str):
        # Extract prediction failure points from the text
        pts = []
        for line in text.splitlines():
            if "Failed prediction at point" in line:
                m = re.search(r"\[([^\]]+)] with true label ([\d.]+)", line)
                if m:
                    nums = [float(x.strip()) for x in m.group(1).split(",")]
                    pts.append({"point": nums, "true_label": float(m.group(2))})
        return pts

    @staticmethod
    def _detect_nested_failure(stdout: str, stderr: str) -> str:
        combined = "\n".join(part for part in (stdout, stderr) if part)
        if not combined.strip():
            return ""
        failure_markers = (
            "Traceback (most recent call last):",
            "NotImplementedError:",
            "ModuleNotFoundError:",
            "FileNotFoundError:",
            "can't open file",
            "CalledProcessError:",
        )
        for marker in failure_markers:
            if marker in combined:
                return combined
        return ""

    @staticmethod
    def _subprocess_output_as_error(stdout: str, stderr: str) -> str:
        stdout = (stdout or "").strip()
        stderr = (stderr or "").strip()
        combined = "\n".join(part for part in (stderr, stdout) if part).strip()
        if not combined:
            return "Subprocess failed with empty output."

        lines = [line.strip() for line in combined.splitlines() if line.strip()]
        for line in reversed(lines):
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*Error: ", line):
                return line
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*Exception: ", line):
                return line
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*: ", line):
                return line

        return combined
