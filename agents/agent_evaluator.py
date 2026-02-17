import os, re, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from entity.code_quality import CodeQuality
from sandbox.executor import execute_code as sandbox_execute_code

class AgentEvaluator:
    """
    Executes the final code with real data and parses AUROC/AUPRC.
    (Logic ported from the old Reviewer.execute_code)
    """

    # ---------- public ----------
    def execute_code(
        self,
        code: str,
        algorithm_name: str,
        package_name: str = "pyod",
        data_files: dict[str, str] | None = None,
    ) -> CodeQuality:
        # Execute the script in sandbox (Modal or local subprocess)
        stdout, stderr, returncode = sandbox_execute_code(
            code=code,
            algorithm_name=algorithm_name,
            package_name=package_name,
            data_files=data_files,
            timeout=180,
        )
        print("\n=== Real-Data Execution Output ===\n", stdout, stderr)

        # If execution failed, return error result
        if returncode != 0:
            return CodeQuality(
                code=code, algorithm=algorithm_name, parameters={}, std_output="",
                error_message=stderr,
                auroc=-1, auprc=-1, error_points=[], review_count=0
            )

        # Parse metrics from the script output
        auroc  = self._find_float(r"AUROC:\s*([\d.]+)", stdout)
        auprc  = self._find_float(r"AUPRC:\s*([\d.]+)", stdout)
        errors = self._parse_errors(stdout)

        # Return evaluation result
        return CodeQuality(
            code=code, algorithm=algorithm_name, parameters={}, std_output=stdout,
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
