"""Check labelled trial integrity and summarize model work by search operator."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def check_trials(directory):
    with (directory / "manifest.csv").open() as stream:
        labels = {str(Path(row["path"]).resolve()): int(row["expected_class"])
                  for row in csv.DictReader(stream) if row["split"] == "development"}
    with (directory / "results.csv").open() as stream:
        trials = list(csv.DictReader(stream))
    with (directory / "protocol.json").open() as stream:
        settings = json.load(stream)["primary"]
    expected = {(image, method) for image in labels for method in ("control", "candidate")}
    seen = set()
    totals = defaultdict(lambda: {"calls": 0, "windows": 0, "windows_with_gain": 0})
    per_image = []
    for row in trials:
        image, method = row["image"], row["method"]
        key = (image, method)
        if key in seen or key not in expected:
            raise ValueError(f"Unexpected or duplicate pair member {key}")
        seen.add(key)
        if (int(row["reference_class"]) != labels[image]
                or int(row["seed"]) != settings["development_seed"]
                or int(row["threads"]) != settings["threads"]
                or row["implementation"] != settings[method]
                or float(row["seconds_limit"]) != settings["seconds"]
                or row["queries_limit"] != str(settings["queries"] or "")):
            raise ValueError(f"Trial context differs from its labelled protocol {key}")
        calls = int(row["queries"])
        if settings["queries"] and calls > settings["queries"]:
            raise ValueError(f"Exceeded call ceiling {key}")
        if float(row["runtime_seconds"]) > settings["seconds"]:
            raise ValueError(f"Exceeded time ceiling {key}")
        metadata = json.loads(row["engine_metadata"])
        windows = metadata.get("windows", [])
        previous_end = None
        row_totals = defaultdict(int)
        for window in windows:
            operator = window["operator"]
            calls = window["end_query"] - window["start_query"]
            if (calls != window["calls"] or not 0 < calls <= metadata["window_calls"]
                    or (previous_end is not None and previous_end != window["start_query"])
                    or window["end_query"] > int(row["queries"])
                    or window["distance"] > window["before_distance"]):
                raise ValueError(f"Inconsistent window trace {key}")
            totals[operator]["calls"] += calls
            totals[operator]["windows"] += 1
            totals[operator]["windows_with_gain"] += window["distance"] < window["before_distance"]
            row_totals[operator] += calls
            previous_end = window["end_query"]
        if method == "candidate":
            per_image.append({"image": Path(image).stem, "status": row["status"],
                              "calls": int(row["queries"]), "runtime_seconds": float(row["runtime_seconds"]),
                              "windows": dict(row_totals),
                              "coarse_capture": metadata.get("coarse_capture"),
                              "coarse_reanchors": metadata.get("coarse_reanchors"),
                              "gains": [w for w in windows if w["distance"] < w["before_distance"]]})
    if seen != expected:
        raise ValueError(f"Missing trial rows: {expected - seen}")
    query_budget = settings["queries"]
    return {"trial_rows": len(trials), "successful_trials": sum(r["status"] == "pass" for r in trials),
            "at_query_limit": sum(int(r["queries"]) == query_budget for r in trials) if query_budget else None,
            "window_totals": dict(totals), "candidate_trials": per_image}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check_trials(args.directory)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: val for key, val in report.items() if key != "candidate_trials"}, indent=2))


if __name__ == "__main__":
    main()
