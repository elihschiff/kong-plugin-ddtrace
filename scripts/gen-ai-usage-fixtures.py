#!/usr/bin/env python3
"""Collect the LLM usage metric conformance cases this plugin implements into one fixture file.

The cases come from libdatadog's libdd-ai-usage crate (libdd-ai-usage/tests/data). Cases that use an
observation field the plugin never fills (canonical model names, rate cards, cost bounds, server tool
counts) are left out.

Usage: gen-ai-usage-fixtures.py <libdd-ai-usage/tests/data> <output.json>
"""

import glob
import json
import os
import sys

UNSUPPORTED_FIELDS = {
    "canonical_model",
    "canonical_provider",
    "model_family",
    "cost_rate_card",
    "cost_bound",
    "cost_priced_as_model",
    "web_search_requests",
    "web_search_requests_secondary",
}
KEPT = (
    "case",
    "profile",
    "observation",
    "deployment_attributes",
    "expected_metrics",
    "expected_resource_attributes",
    "expected_error_code",
    "historical",
    "points",
    "resource_attributes",
    "expected_series",
)


def load(path):
    with open(path) as f:
        case = json.load(f)
    return {key: case[key] for key in KEPT if key in case}


def main(data, output):
    fixtures = {"projection": [], "datadog": [], "skipped": 0}
    for directory in ("provider-attempt", "provider-token-breakdown", "gateway-request"):
        for path in sorted(glob.glob(os.path.join(data, directory, "*", "*.json"))):
            case = load(path)
            observation = case.get("observation")
            if isinstance(observation, dict) and UNSUPPORTED_FIELDS & {
                key for key, value in observation.items() if value is not None
            }:
                fixtures["skipped"] += 1
                continue
            fixtures["projection"].append(case)
    for path in sorted(glob.glob(os.path.join(data, "datadog", "*.json"))):
        fixtures["datadog"].append(load(path))
    with open(output, "w") as f:
        json.dump(fixtures, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
