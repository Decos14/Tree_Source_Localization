import argparse
import json
from typing import Optional
from .Tree import Tree


def main():
    parser = argparse.ArgumentParser(description="Run source localization on a tree.")
    parser.add_argument(
        "--tree_path",
        type=str,
        help="Path to the JSON file of the tree.")
    parser.add_argument(
        "--observers", 
        nargs='+',
        help="List of observer node names (space-separated).")
    parser.add_argument(
        "--infection_times",
        type=str,
        help='Path to the JSON file of infection times.'
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["linear", "exponential", "exact"],
        default=None,
        help="Optional augmentation method."
    )

    args = parser.parse_args()

    # Prompt for any missing arguments
    if args.tree_path is None:
        tree_path = input("Enter path to tree JSON file: ").strip()
    else:
        tree_path = args.tree_path

    if args.observers is None:
        observers_input = input("Enter observer names (space-separated): ").strip()
        observers = observers_input.split()
    else:
        observers = args.observers

    if args.infection_times is None:
        infection_path = input("Enter path to the infection times JSON file: ").strip()
    else:
        infection_path = args.infection_times
    with open(infection_path, 'r') as f:
        infection_times = json.load(infection_path)

    # Optional method
    method = args.method
    if method is None:
        method_input = input("Enter method (linear, exponential, exact) or leave blank: ").strip()
        method = method_input if method_input else None

    # Run localization
    tree = Tree(
        file_name=tree_path,
        observers=observers,
        infection_times=infection_times
    )
    source = tree.localize(method=method)
    print(f"\nMost likely source node: {source}")