from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError


def hub_file_exists(repo_id: str, filename: str, revision: str = None, repo_type: str = None) -> bool:
    r"""
    Checks if a file exists in a remote Hub repository.
    """
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
    try:
        get_hf_file_metadata(url)
        return True
    except EntryNotFoundError:
        return False
