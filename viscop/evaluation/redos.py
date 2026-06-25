import json
import re
import argparse
import sys
import os
from collections import defaultdict
from tqdm import tqdm

# Try to import PrettyTable (used in your original show_metrics)
try:
    from prettytable import PrettyTable
except ImportError:
    print("Warning: 'prettytable' library not found. Output will be simple text.")
    PrettyTable = None

try:
    import llm_answer_parsing
except ImportError:
    print("Error: Could not import 'llm_answer_parsing'. Make sure the file is in the current directory.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Fix -1 predictions and calculate detailed metrics.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .json file")
    parser.add_argument("--output_file", type=str, default="fixed_output.json", help="Path to save new .json file")
    return parser.parse_args()

def show_metrics(metrics, benchmark="EgoPerception"):
    """
    Prettifies the output table exactly like your original script.
    """
    if PrettyTable and all(isinstance(metric, (int, float)) for metric in metrics.values()):
        table = PrettyTable(["Task Type", "Accuracy"])
        # Sort keys to make 'Overall' appear last or first consistently
        keys = sorted(metrics.keys())
        if "Overall" in keys:
            keys.remove("Overall")
            keys.append("Overall")
            
        for task_name in keys:
            metric = metrics[task_name]
            table.add_row([task_name, round(metric, 2)])
        
        table.align["Task Type"] = "l"
        print(f"\nResults on {benchmark}:")
        print(table)
        print("\n")
    else:
        # Fallback if prettytable is missing
        print(f"\nResults on {benchmark}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")

def calculate_detailed_metrics(data):
    """
    Replicates the logic of _eval_mcqa to breakdown accuracy by task_type.
    """
    samples = defaultdict(list)
    overall_samples = []

    # Skip header if it exists
    start_idx = 0
    if len(data) > 0 and "matching" not in data[0]:
        start_idx = 1

    for i in range(start_idx, len(data)):
        item = data[i]
        
        # Ensure we have necessary keys
        if "matching" not in item or "ground_truth" not in item:
            continue

        matching = int(item["matching"])
        overall_samples.append(matching)

        # Get task types (handle list or string)
        # We check top-level first, then meta_data
        task_types = item.get("task_type")
        if not task_types and "meta_data" in item:
            task_types = item["meta_data"].get("task_type")

        if task_types:
            if isinstance(task_types, (list, tuple)):
                for t in task_types:
                    samples[t].append(matching)
            else:
                samples[task_types].append(matching)

    # Calculate percentages
    metrics = {k: sum(v) / len(v) * 100 for k, v in samples.items()}
    
    if overall_samples:
        metrics["Overall"] = sum(overall_samples) / len(overall_samples) * 100
    else:
        metrics["Overall"] = 0.0

    return metrics

def process_single_item(item):
    """
    Fixes a single item with prediction == -1
    """
    meta = item.get("meta_data", {})
    response = item.get("response", "")
    response_clean = response.replace('answer', '').replace('Answer', '')
    
    question = meta.get("question", "")
    options = meta.get("options", [])
    option_letters = meta.get("option_letters", [])
    
    if not options or not option_letters:
        return -1, False

    # 1. Build Prompt and Call LLM
    choices_str = " ".join(f'({l}) {o}' for l, o in zip(option_letters, options))
    prompt = llm_answer_parsing.build_prompt(question, choices_str, response_clean)
    
    try:
        llm_response = llm_answer_parsing.parse_with_llama(prompt)
    except Exception:
        return -1, False

    # 2. Fix: Check LLM output
    max_letter = sorted(option_letters)[-1]
    pred_matches = re.findall(f'[\(\ ]*([A-{max_letter}])[ \)\.]*', llm_response)

    pred_idx = -1
    found_flag = False

    if pred_matches:
        letter_found = pred_matches[-1].strip()
        if letter_found in option_letters:
            pred_idx = option_letters.index(letter_found)
            found_flag = True

    # 3. Fallback: Check original text
    if not found_flag:
        for idx, opt in enumerate(options):
            opt_clean = opt.strip().strip('.')
            if opt_clean.lower() in response.lower():
                pred_idx = idx
                found_flag = True
                break
    
    return pred_idx, found_flag

def main():
    args = parse_args()

    print(f"Loading {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    output_file = args.input_file.replace('.json', '_fixed.json')

    # Handle summary dict at index 0
    start_idx = 0
    if len(data) > 0 and "matching" not in data[0]:
        start_idx = 1

    fixed_count = 0
    
    print(f"Scanning {len(data) - start_idx} items...")
    
    for i in tqdm(range(start_idx, len(data))):
        item = data[i]
        
        # Only process broken predictions
        if item.get("prediction") == -1:
            new_pred, success = process_single_item(item)
            if success:
                item["prediction"] = new_pred
                item["matching"] = (new_pred == item["ground_truth"])
                item["post_processed"] = True 
                fixed_count += 1

    print("-" * 30)
    print(f"Fixed {fixed_count} broken predictions.")
    
    # --- METRICS CALCULATION AND DISPLAY ---
    metrics = calculate_detailed_metrics(data)
    
    # Update the summary dictionary at index 0 if it exists, otherwise insert it
    if start_idx == 1:
        data[0] = metrics
    else:
        data.insert(0, metrics)

    show_metrics(metrics)

    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()