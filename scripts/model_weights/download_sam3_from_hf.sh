#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <hf_repo_id> <output_dir> [revision]" >&2
  echo "Example: $0 user/simfoundry-sam3-weights /srv/simfoundry-models" >&2
  exit 2
fi

repo_id="$1"
out_dir="$2"
revision="${3:-main}"

mkdir -p "$out_dir"
hf download "$repo_id" sam3.pt LICENSE \
  --revision "$revision" \
  --local-dir "$out_dir"

test -s "$out_dir/sam3.pt"
test -s "$out_dir/LICENSE"
echo "SAM3 weights ready at: $out_dir/sam3.pt"
