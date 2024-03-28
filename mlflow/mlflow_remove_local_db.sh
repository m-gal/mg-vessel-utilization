#!/usr/bin/env sh

echo "Permanently remove mlruns and mlregistry db..."

# Blow away mlruns artifact direcotory and sqlite db file

/bin/rm -f -r mlruns && /bin/rm -f -r mlregistry.db
