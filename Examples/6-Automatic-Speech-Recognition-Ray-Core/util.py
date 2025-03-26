from pathlib import Path
import mlflow
import shutil
import os


def copy_folder_recursively(source: str, destination: str):
    # Create the destination directory if it does not exist
    os.makedirs(destination, exist_ok=True)
    
    for item in os.listdir(source):
        source_item = os.path.join(source, item)
        destination_item = os.path.join(destination, item)
        
        if os.path.isdir(source_item):
            # If the item is a directory, call the function recursively
            copy_folder_recursively(source_item, destination_item)
        else:
            # If the item is a file, copy it to the destination
            shutil.copy2(source_item, destination_item)

def stage_registered_model(
  *,
  catalog: str, 
  schema: str, 
  model_name: str,
  local_base_path: str,
  version: int = None,
  alias: str = None,
  overwrite: bool = False
) -> Path:
  
  if alias is not None:
    model_uri = f"models:/{catalog}.{schema}.{model_name}@{alias}"
    model_folder_name = alias

  else:
    if version is None:
      version = 1 # Defaults to 1
    model_folder_name = str(version)
    model_uri = f"models:/{catalog}.{schema}.{model_name}/{version}"

  local_base_path = Path(local_base_path)
  local_model_path = local_base_path / catalog / schema / model_name / str(model_folder_name)   
  if overwrite is False and local_model_path.exists() and any(local_model_path.iterdir()):
    print(f"Model path {local_model_path} already exists and contains content.")
    return local_model_path
  if overwrite is True and local_model_path.exists():
    shutil.rmtree(str(local_model_path))
    print(f"Existing model path {local_model_path} removed.")

  if str(local_base_path).startswith("/dbfs") or str(local_base_path).startswith("/Volumes"):
    local_disk_path = "/local_disk0" + str(local_model_path)
    if overwrite is True and Path(local_disk_path).exists():
      shutil.rmtree(local_disk_path)
      print(f"Existing model path in local disk {local_disk_path} removed.")
    mlflow.artifacts.download_artifacts(model_uri, dst_path=local_disk_path)
    print(f"Model artifacts downloaded to local disk {local_disk_path}")
    copy_folder_recursively(source=str(local_disk_path), destination=str(local_model_path))
    print(f"Model artifacts copied to dbfs or volume {local_model_path}")
  else:
    mlflow.artifacts.download_artifacts(model_uri, dst_path=local_model_path)
  print(f"Model from uc registry saved in: {local_model_path}")
  return local_model_path

def flatten_folder(root_folder):
    """
    Moves all files from subdirectories (including nested ones) to the root folder.
    If a filename conflict occurs, a numeric suffix is added to the filename.
    After moving files, empty directories are removed.
    """
    # Walk the directory tree from bottom to top to safely remove empty directories.
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        # Skip the root folder since we want to move files into it.
        if os.path.abspath(dirpath) == os.path.abspath(root_folder):
            continue

        for filename in filenames:
            source_file = os.path.join(dirpath, filename)
            destination_file = os.path.join(root_folder, filename)

            # Handle filename conflicts by appending a counter.
            if os.path.exists(destination_file):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(destination_file):
                    destination_file = os.path.join(root_folder, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(source_file, destination_file)
            print(f"Moved: {source_file} --> {destination_file}")

        # Attempt to remove the directory if it is empty.
        try:
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")
        except OSError as e:
            print(f"Could not remove {dirpath}: {e}")


def force_on_node(node_id: str, remote_func_or_actor_class):
    import ray

    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=False
    )
    options = {"scheduling_strategy": scheduling_strategy}
    return remote_func_or_actor_class.options(**options)


def run_on_every_node(remote_func_or_actor_class, **remote_kwargs):
    import ray

    refs = []
    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("GPU", None):
            refs.append(
                force_on_node(node["NodeID"], remote_func_or_actor_class).remote(
                    **remote_kwargs
                )
            )
    return ray.get(refs)