#!/usr/bin/env bash
# KAt-DataCenter (Paderborn): download sequencial e conferencia pelo sha256 do manifest.
# Licenca CC BY-NC 4.0; dado bruto fora do git. Uso: paderborn_download.sh <destino> <paderborn.sha256>
# Extracao depois: uv run --no-project --with unrar-cffi python -I extract_rar.py <destino>/archives <destino>
set -euo pipefail
D="$1"; SUMS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
A="$D/archives"; U="https://groups.uni-paderborn.de/kat/BearingDataCenter"
mkdir -p "$A"
for f in $(awk '{print $2}' "$SUMS"); do
  wget -q -c --tries=5 --waitretry=10 -O "$A/$f" "$U/$f" && echo "ok $f" || echo "FALHOU $f"
done
( cd "$A" && sha256sum -c "$SUMS" )
