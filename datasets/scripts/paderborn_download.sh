#!/usr/bin/env bash
# KAt-DataCenter (Paderborn): sequential download, checked against the manifest's sha256.
# License CC BY-NC 4.0; raw data outside git. Usage: paderborn_download.sh <target> <paderborn.sha256>
# Extraction afterwards: uv run --no-project --with unrar-cffi python -I extract_rar.py <target>/archives <target>
set -euo pipefail
D="$1"; SUMS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
A="$D/archives"; U="https://groups.uni-paderborn.de/kat/BearingDataCenter"
mkdir -p "$A"
for f in $(awk '{print $2}' "$SUMS"); do
  wget -q -c --tries=5 --waitretry=10 -O "$A/$f" "$U/$f" && echo "ok $f" || echo "FAILED $f"
done
( cd "$A" && sha256sum -c "$SUMS" )
