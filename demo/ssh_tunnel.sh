#!/usr/bin/env bash
# Forward the demo ports of a remote GPU server to this machine over SSH, so the
# Gradio app running there (python app.py) opens as http://localhost:<port>.
#
#   demo/ssh_tunnel.sh user@server            # forwards 7860 (the app's default port)
#   demo/ssh_tunnel.sh user@server 7863 7861  # forwards several ports
#   UMUST_SSH_HOST=user@server demo/ssh_tunnel.sh
#
# Extra ssh options can be passed through SSH_OPTS, e.g. SSH_OPTS="-p 2222 -i ~/.ssh/lab".
# The tunnel stays up until Ctrl-C; keepalives prevent idle disconnects.
set -euo pipefail

HOST="${1:-${UMUST_SSH_HOST:-}}"
if [ -z "$HOST" ]; then
  echo "usage: $0 user@server [port ...]   (or set UMUST_SSH_HOST)" >&2
  exit 1
fi
shift || true
PORTS=("$@")
[ ${#PORTS[@]} -eq 0 ] && PORTS=(7860)

FORWARDS=()
for p in "${PORTS[@]}"; do
  FORWARDS+=(-L "${p}:localhost:${p}")
done

echo "Forwarding from $HOST:"
for p in "${PORTS[@]}"; do
  echo "  http://localhost:${p}"
done
echo "Press Ctrl-C to close the tunnel."

# -N: no remote command; ExitOnForwardFailure: fail loudly if a local port is taken
# shellcheck disable=SC2086
exec ssh -N \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  ${SSH_OPTS:-} "${FORWARDS[@]}" "$HOST"
