#!/usr/bin/env bash
set -euo pipefail

destination=${1:?usage: download_genept_assets.sh DESTINATION}
archive="$destination/GenePT_emebdding_v2.zip"
partial="$archive.partial"
url=https://zenodo.org/api/records/10833191/files/GenePT_emebdding_v2.zip/content
expected_md5=3f6ce4317e3a0091978ae5cb8fbf05a3

mkdir -p "$destination"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 5 --continue-at - --output "$partial" "$url"
  mv "$partial" "$archive"
fi

actual_md5=$(md5sum "$archive" | awk '{print $1}')
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "GenePT archive checksum mismatch: $actual_md5" >&2
  exit 1
fi

unzip -oq "$archive" -d "$destination"
find "$destination" -maxdepth 2 -type f -printf '%P\t%s bytes\n' | sort
