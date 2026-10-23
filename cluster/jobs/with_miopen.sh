#!/bin/bash
# Stage the MIOpen Find DB onto node-local /tmp, run the command, copy it back.
# Lustre cannot lock the DB. MIOPEN_USER_DB_PATH and MIOPEN_CACHE are set by lumi_ddp.sh.
set -euo pipefail
: "${MIOPEN_USER_DB_PATH:?}"
: "${MIOPEN_CACHE:?}"

mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CACHE}"
if [[ -n "$(ls -A "${MIOPEN_CACHE}" 2>/dev/null || true)" ]]; then
  cp -a "${MIOPEN_CACHE}/." "${MIOPEN_USER_DB_PATH}/"
fi

save() {
  cp -a "${MIOPEN_USER_DB_PATH}/." "${MIOPEN_CACHE}/" || true
  echo "MIOPEN save ${MIOPEN_USER_DB_PATH} -> ${MIOPEN_CACHE}"
}
trap save EXIT
exec "$@"
