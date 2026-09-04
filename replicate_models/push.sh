#!/usr/bin/env bash
# Build and push the four Replicate models (one per task) with cog.
#
#   export REPLICATE_API_TOKEN=r8_...          # token of the owning account/org
#   export HF_TOKEN=hf_...                     # can read the weight repository
#   replicate_models/push.sh [owner] [task ...] [--hardware gpu-l40s] [--test]
#
# Defaults: owner=malerlab, all four tasks, hardware gpu-l40s. Creates the
# Replicate models when missing (private), then `cog push`. --test runs a
# local `cog predict` smoke test on the example score before pushing.
# UMUST_HF_WEIGHTS_REPO overrides the weight repository baked into the image
# (default MALer-Lab/u-must-i2a-piano; malerlab/u-must for the gated release).
#
# COG_BIN selects the cog CLI. The default prefers a legacy 0.16.x binary
# (installed as `cog-0.16`) because the 0.17+ runtime does not upload files
# nested inside a BaseModel output — they come back as data URIs that
# Replicate does not store:
#   curl -L -o ~/.local/bin/cog-0.16 https://github.com/replicate/cog/releases/download/v0.16.12/cog_$(uname -s)_$(uname -m)
set -euo pipefail
COG_BIN="${COG_BIN:-$(command -v cog-0.16 || command -v cog)}"
echo "using $($COG_BIN --version)"
cd "$(dirname "$0")/.."

OWNER="malerlab"; HARDWARE="gpu-l40s"; TEST=0; TASKS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --hardware) HARDWARE="$2"; shift 2;;
    --test) TEST=1; shift;;
    omr|midi-to-audio|image-to-audio|contin-u) TASKS+=("$1"); shift;;
    *) OWNER="$1"; shift;;
  esac
done
[ ${#TASKS[@]} -eq 0 ] && TASKS=(omr midi-to-audio image-to-audio contin-u)
: "${REPLICATE_API_TOKEN:?set REPLICATE_API_TOKEN}"
: "${HF_TOKEN:?set HF_TOKEN (read access to the weight repository)}"

SECRET_FILE=$(mktemp); trap 'rm -f "$SECRET_FILE"' EXIT
printf '%s' "$HF_TOKEN" > "$SECRET_FILE"
# r8.im accepts an API token as the registry password (cog login wants the
# separate CLI token from replicate.com/auth/token instead)
docker login r8.im -u "$OWNER" --password-stdin <<< "$REPLICATE_API_TOKEN" >/dev/null

ensure_model() {  # create the Replicate model if it does not exist
  local name="$1"
  if curl -sf -H "Authorization: Bearer $REPLICATE_API_TOKEN" "https://api.replicate.com/v1/models/$OWNER/$name" >/dev/null; then
    return
  fi
  echo "Creating $OWNER/$name ($HARDWARE)"
  # Creating a private model through the API currently fails with a 500 for
  # organizations, so create it public and switch the visibility right after.
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Authorization: Bearer $REPLICATE_API_TOKEN" -H "Content-Type: application/json" \
    https://api.replicate.com/v1/models \
    -d "{\"owner\":\"$OWNER\",\"name\":\"$name\",\"visibility\":\"private\",\"hardware\":\"$HARDWARE\"}")
  if [ "$code" != "201" ]; then
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Authorization: Bearer $REPLICATE_API_TOKEN" -H "Content-Type: application/json" \
      https://api.replicate.com/v1/models \
      -d "{\"owner\":\"$OWNER\",\"name\":\"$name\",\"visibility\":\"public\",\"hardware\":\"$HARDWARE\"}")
    [ "$code" = "201" ] || { echo "could not create $OWNER/$name (HTTP $code)"; exit 1; }
    curl -s -o /dev/null -X PATCH -H "Authorization: Bearer $REPLICATE_API_TOKEN" -H "Content-Type: application/json" \
      "https://api.replicate.com/v1/models/$OWNER/$name" -d '{"visibility":"private"}' || true
  fi
  echo "created https://replicate.com/$OWNER/$name"
}

for task in "${TASKS[@]}"; do
  name="u-must-$task"
  ensure_model "$name"
  echo "== $OWNER/$name  (cog.$task.yaml)"
  if [ "$TEST" = 1 ]; then
    "$COG_BIN" build -f "cog.$task.yaml" -t "u-must-$task:test" --secret "id=hf_token,src=$SECRET_FILE"
    case "$task" in
      omr)            "$COG_BIN" predict -f "cog.$task.yaml" "u-must-$task:test" -i image=@demo/examples/bach_bwv846_prelude_page1.png -i system=1 ;;
      midi-to-audio)  "$COG_BIN" predict -f "cog.$task.yaml" "u-must-$task:test" -i midi=@demo/examples/bach_bwv846_prelude.mid -i max_duration_sec=20 ;;
      image-to-audio) "$COG_BIN" predict -f "cog.$task.yaml" "u-must-$task:test" -i image=@demo/examples/bach_bwv846_prelude_page1.png -i system=1 ;;
      contin-u)       "$COG_BIN" predict -f "cog.$task.yaml" "u-must-$task:test" -i score=@demo/examples/bach_bwv846_prelude.pdf -i last_page=1 -i max_systems=3 ;;
    esac
  fi
  "$COG_BIN" push -f "cog.$task.yaml" "r8.im/$OWNER/$name" --secret "id=hf_token,src=$SECRET_FILE"
done
