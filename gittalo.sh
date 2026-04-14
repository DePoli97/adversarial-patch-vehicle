#!/bin/bash
EXPECTED_AUTHOR="DePoli97"
CURRENT_AUTHOR=$(git config user.name)
if [ "$CURRENT_AUTHOR" != "$EXPECTED_AUTHOR" ]; then
  echo "Aborting: git user.name is '$CURRENT_AUTHOR', expected '$EXPECTED_AUTHOR'."
  echo "Fix with: git config user.name \"$EXPECTED_AUTHOR\""
  exit 1
fi
git pull
git add .
git commit -m "auto remote machine synchronization"
git push
