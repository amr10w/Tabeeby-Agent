import os
from .helpers import is_safe_path,WORKSPACE_DIR
  
def list_files(path:str =".") -> str:
    """List files and directories inside a workspace path.

    Returns a newline-delimited list of entries, marking directories with '/'.
    """

    try:

        safe_target=is_safe_path(path)

        if not safe_target.exists():
            return f"Error: Path '{path}' does not exist."

        elif not safe_target.is_dir():
            return f"Error: Path '{path}' is a file, not a directory."

        items=[]
        for item in sorted(safe_target.iterdir()):
            suffix="/" if item.is_dir() else ""
            items.append(f"{item.name}{suffix}".strip())

        if not items:
            return f"Directory '{path}' is empty."
        
        return "\n".join(items)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def read_file(path: str) -> str:
    """Read and return the text contents of a workspace file.

    The path is validated to remain inside the repository sandbox.
    """
    try:
        safe_target = is_safe_path(path)
        if not safe_target.exists():
            return f"Error: File '{path}' does not exist."
        return safe_target.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

def write_file(path: str, content: str) -> str:
    """Write text content to a workspace file.

    Creates parent directories if needed and ensures the target remains inside
    the workspace sandbox.
    """
    try:
        safe_target = is_safe_path(path)
        # Create folder if path is nested (e.g. my-site/index.html)
        safe_target.parent.mkdir(parents=True, exist_ok=True)
        safe_target.write_text(content, encoding="utf-8")
        return f"Successfully wrote to '{path}'."
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"



def create_directory(path: str) -> str:
    """Create a directory (and parents) inside the working directory.

    Args:
        path: Relative directory path to create.

    Returns:
        A human-readable success message.

    Raises:
        ValueError: If `path` fails validation performed by `validate_path`.
    """
    clean_path = is_safe_path(path=path)
    os.makedirs(clean_path, exist_ok=True)
    return f"Directory '{clean_path}' created successfully."

