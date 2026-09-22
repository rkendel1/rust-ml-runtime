#!/bin/sh
set -eu

repository="${ML_RUNTIME_REPOSITORY:-rkendel1/rust-ml-runtime}"
version="${ML_RUNTIME_VERSION:-}"
install_dir="${ML_RUNTIME_INSTALL_DIR:-${HOME}/.local/bin}"

command -v curl >/dev/null 2>&1 || { echo 'ml-runtime installer requires curl' >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo 'ml-runtime installer requires tar' >&2; exit 1; }

if [ -z "$version" ]; then
  latest_url="$(curl --fail --silent --show-error --location --output /dev/null --write-out '%{url_effective}' "https://github.com/${repository}/releases/latest")"
  version="${latest_url##*/v}"
fi
case "$version" in v*) version="${version#v}" ;; esac

case "$(uname -s)" in
  Darwin) platform=macos ;;
  Linux) platform=linux ;;
  *) echo "unsupported operating system: $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  arm64|aarch64) architecture=aarch64 ;;
  x86_64|amd64) architecture=x86_64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

artifact="ml-runtime-v${version}-${platform}-${architecture}.tar.gz"
base="${ML_RUNTIME_RELEASE_BASE:-https://github.com/${repository}/releases/download/v${version}}"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT HUP INT TERM
curl --fail --silent --show-error --location "${base}/${artifact}" --output "${temporary}/${artifact}"
curl --fail --silent --show-error --location "${base}/${artifact}.sha256" --output "${temporary}/${artifact}.sha256"
expected="$(awk 'NR == 1 { print $1 }' "${temporary}/${artifact}.sha256")"
case "$expected" in ''|*[!0-9a-fA-F]*) echo 'invalid checksum file' >&2; exit 1 ;; esac
[ "${#expected}" -eq 64 ] || { echo 'invalid checksum length' >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "${temporary}/${artifact}" | awk '{ print $1 }')"
else
  actual="$(shasum -a 256 "${temporary}/${artifact}" | awk '{ print $1 }')"
fi
[ "$actual" = "$expected" ] || { echo "checksum verification failed for ${artifact}" >&2; exit 1; }

tar -xzf "${temporary}/${artifact}" -C "$temporary"
binary="${temporary}/ml-runtime-v${version}-${platform}-${architecture}/ml-runtime"
[ -f "$binary" ] || { echo 'verified archive does not contain ml-runtime' >&2; exit 1; }
mkdir -p "$install_dir"
cp "$binary" "${install_dir}/ml-runtime"
chmod 755 "${install_dir}/ml-runtime"
echo "Installed ml-runtime ${version} to ${install_dir}/ml-runtime"
case ":${PATH:-}:" in
  *":${install_dir}:"*) ;;
  *) echo "Add ${install_dir} to PATH to run ml-runtime from any shell." ;;
esac
