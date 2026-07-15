import os
import shutil
from pathlib import Path

def main():
    base_dir = Path("/home/ruben-eschauzier/projects/process-caching-journal")
    random_new_dir = base_dir / "data" / "random_new"
    random_dir = base_dir / "data" / "random"
    random_bak_dir = base_dir / "data_bak" / "random"
    
    # 1. Restore original data/random files from data_bak/random
    print(f"Restoring original files in {random_dir} from {random_bak_dir}...")
    if random_bak_dir.exists():
        # Clear current files in random_dir
        for f in random_dir.glob("query-results-raw-*.json"):
            try:
                f.unlink()
            except Exception as e:
                print(f"Error removing {f}: {e}")
        # Copy from backup
        for f in random_bak_dir.glob("query-results-raw-*.json"):
            shutil.copy2(f, random_dir / f.name)
        print("Restoration complete.")
    else:
        print("Warning: Backup directory data_bak/random not found. Skipping restoration.")
        
    # Ensure destination directory random_new exists
    random_new_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning directories in {random_new_dir}...")
    
    # Iterate through all subdirectories in random_new
    for algo_path in random_new_dir.iterdir():
        if not algo_path.is_dir():
            continue
        
        algo_name = algo_path.name
        print(f"Processing algorithm: {algo_name}")
        
        # Look for combination directories
        for combo_path in algo_path.iterdir():
            if not combo_path.is_dir():
                continue
            
            # Check if name is like combination_* or combinations_*
            combo_name = combo_path.name
            if not (combo_name.startswith("combination_") or combo_name.startswith("combinations_")):
                continue
            
            # Extract the index number
            try:
                combo_idx = int(combo_name.split("_")[-1])
            except ValueError:
                print(f"Skipping directory with invalid format: {combo_path}")
                continue
            
            # Locate query-results-raw.json
            raw_json_path = combo_path / "query-results-raw.json"
            if not raw_json_path.exists():
                print(f"Warning: {raw_json_path} does not exist.")
                continue
            
            # Determine destination filename
            if algo_name == "default":
                if combo_idx == 0:
                    dest_name = "query-results-raw-default.json"
                else:
                    print(f"Warning: Found default with unexpected combination index {combo_idx}")
                    continue
            else:
                # Map 0 -> s, 1 -> m, 2 -> l
                size_map = {0: "s", 1: "m", 2: "l"}
                if combo_idx in size_map:
                    size_suffix = size_map[combo_idx]
                else:
                    print(f"Warning: Unexpected combination index {combo_idx} for {algo_name}")
                    continue
                
                dest_name = f"query-results-raw-{algo_name}-{size_suffix}.json"
            
            dest_path = random_new_dir / dest_name
            print(f"Copying {raw_json_path.relative_to(base_dir)} to {dest_path.relative_to(base_dir)}")
            shutil.copy2(raw_json_path, dest_path)
            
    print("Processing complete!")

if __name__ == "__main__":
    main()
