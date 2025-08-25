#!/usr/bin/env bash

if command -v docker-compose >/dev/null; then
  cmd="docker-compose"
elif command -v docker compose >/dev/null; then
  cmd="docker compose"
else
  echo "Docker compose missed..."
  exit 1
fi

echo "RENDER_GID=$(getent group render | cut -d: -f3)" > .env
echo "VIDEO_GID=$(getent group video  | cut -d: -f3)" >> .env

if [[ $1 == "up" ]]
then
    echo "Docker compose deploy"
    $cmd up -d
else
    if [[ $1 == "down" ]]
    then
        echo "Docker compose destroy"
        $cmd down --remove-orphans -v
    fi
fi
