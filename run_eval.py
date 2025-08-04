import os
from pathlib import Path # <-- IMPORT THE PATH OBJECT
os.environ["OPENAI_API_KEY"] = "ollama_is_local"

from evals.registry import Registry
from evals.record import LocalRecorder, RunSpec
from custom_models.my_ollama_model import OllamaModel

def main():
    print("Initializing...")

    # --- Configuration ---
    model_name = "mistral"
    eval_name = "math_eval.v1"
    model_string_id = f"custom_models.my_ollama_model/{model_name}"
    log_path = "./run_results.jsonl"
    # ---------------------

    project_root = os.path.dirname(os.path.abspath(__file__))
    model = OllamaModel(model=model_name)
    registry = Registry(registry_paths=[project_root])

    run_spec = RunSpec(
        completion_fns=[model_string_id],
        eval_name=eval_name,
        base_eval="math_eval",
        split="v1",
        run_id=f"{model_name}_{eval_name}_{os.urandom(4).hex()}",
        run_config={},
        created_by="local-user"
    )

    recorder = LocalRecorder(log_path=log_path, run_spec=run_spec)

    print(f"Fetching spec for eval: {eval_name}")
    eval_spec = registry.get_eval(eval_name)
    eval_class = registry.get_class(eval_spec)

    # Create an instance of the evaluation class
    eval_instance = eval_class(
        completion_fns=[model],
        # CORRECTED: Convert the project_root string to a Path object
        eval_registry_path=Path(project_root),
        **eval_spec.args
    )

    print("Running evaluation...")
    eval_instance.run(recorder)

    print("\n" + "="*40)
    print("🎉 Evaluation finished successfully! 🎉")
    print(f"Results have been saved in: {log_path}")
    print("="*40)

if __name__ == "__main__":
    main()