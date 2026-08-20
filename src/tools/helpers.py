from pathlib import Path

WORKSPACE_DIR = (Path(__file__).parent.parent / "memory").resolve()

def is_safe_path(name:str, base_dir=WORKSPACE_DIR) -> Path:
    """Resolve and validate a workspace-relative path.

    Blocks path traversal attacks such as '../../etc' so tools cannot escape
    the workspace sandbox.
    """

    base_dir.mkdir(exist_ok=True)
    target = (base_dir/name).resolve()

    if not target.is_relative_to(base_dir.resolve()):
        raise ValueError(f"Blocked: '{name}' is outside the workspace sandbox.")
    
    return target

