#!/bin/bash
CURRENT_AUTHOR=$(git config user.name)

if [[ "$CURRENT_AUTHOR" != "Paolo Deidda" && "$CURRENT_AUTHOR" != "DePoli97" ]]; then
  echo "Aborting: git user.name is '$CURRENT_AUTHOR', expected 'Paolo Deidda' or 'DePoli97'."
  echo "Fix with: git config user.name \"Paolo Deidda\""
  exit 1
fi

# Concatenare i comandi con && è più sicuro e conciso: se uno fallisce, i successivi si fermano
git pull && git add . && git commit -m "home - quick repo synchronization" && git push
