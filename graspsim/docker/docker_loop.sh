#!/bin/bash

set -e

# Variablen für Omni-Konfiguration
command="$@"
if [[ -z "$@" ]]; then
    command="bash"
fi

omni_server="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"
if ! [[ -z "${OMNI_SERVER}" ]]; then
    omni_server="${OMNI_SERVER}"
fi

omni_user="karaadem"
if ! [[ -z "${OMNI_USER}" ]]; then
    omni_user="${OMNI_USER}"
fi

omni_password="karaadem"
if ! [[ -z "${OMNI_PASS}" ]]; then
    omni_password="${OMNI_PASS}"
fi

accept_eula=""
if ! [[ -z "${ACCEPT_EULA}" ]]; then
    accept_eula="${ACCEPT_EULA}"
fi

privacy_consent=""
if ! [[ -z "${PRIVACY_CONSENT}" ]]; then
    privacy_consent="${PRIVACY_CONSENT}"
fi

privacy_userid="${omni_user}"
if ! [[ -z "${PRIVACY_USERID}" ]]; then
    privacy_userid="${PRIVACY_USERID}"
fi

# Weitere Pfade
graspsim_path="/home/karaadem/git/OptiSim/graspsim"
scene_path="/mnt/4TBSSD/synthetic_data/share"

# Docker- und Skript-Konfiguration
CONTAINER_NAME="graspsim_container2"
IMAGE_NAME="graspsim"
WORKDIR="/graspsim"  # Arbeitsverzeichnis im Container
PYTHON_EXEC="omni_python"  # Python-Interpreter oder Wrapper
PYTHON_SCRIPT="graspsim.py"  # Python-Skript
PYTHON_PARAMS='-i "25-50" -r robot_configs/config.toml'  # Parameter für das Python-Skript
RETRY_LIMIT=10  # Maximale Anzahl von Neustart-Versuchen
RETRY_COUNT=0  # Aktueller Zähler von Neustart-Versuchen

# Docker-Container starten
start_container() {
    echo "Starte Docker-Container mit X11-Forwarding im Hintergrund..."
    xhost +
    docker run --name "${CONTAINER_NAME}" --entrypoint /bin/sh -d --gpus "device=1" -e "ACCEPT_EULA=${accept_eula}" --rm --network=host \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -e DISPLAY \
        -e "OMNI_USER=${omni_user}" -e "OMNI_PASS=${omni_password}" \
        -e "OMNI_SERVER=${omni_server}" \
        -e "PRIVACY_CONSENT=${privacy_consent}" -e "PRIVACY_USERID=${privacy_userid}" \
        -v ~/docker/isaac-sim/kit/cache/Kit:/isaac-sim/kit/cache:rw \
        -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
        -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/documents:/root/Documents:rw \
        -v "${graspsim_path}:/graspsim:rw" \
        -v "${scene_path}:/share:rw" \
        "${IMAGE_NAME}" -c "tail -f /dev/null"
}

# Container bereinigen und stoppen
cleanup_container() {
    echo "Bereinige alten Container..."
    docker rm -f "${CONTAINER_NAME}" || true
}

# Hauptlogik
while (( RETRY_COUNT < RETRY_LIMIT )); do
    # Prüfe, ob der Container läuft
    if ! docker ps --filter "name=${CONTAINER_NAME}" | grep -q "${CONTAINER_NAME}"; then
        cleanup_container
        start_container
    fi

    # Führe das Python-Skript im angegebenen Pfad aus
    echo "Wechsel in das Arbeitsverzeichnis: ${WORKDIR}"
    echo "Führe Python-Skript aus: ${PYTHON_EXEC} ${PYTHON_SCRIPT} ${PYTHON_PARAMS}"
    docker exec "${CONTAINER_NAME}" bash -c "cd ${WORKDIR} && \$${PYTHON_EXEC} ${PYTHON_SCRIPT} ${PYTHON_PARAMS}"

    # Überprüfen, ob das Skript erfolgreich war
    if [ $? -eq 0 ]; then
        echo "Python-Skript erfolgreich ausgeführt."
        break
    else
        echo "Fehler erkannt. Container wird neu gestartet..."
        cleanup_container
        ((RETRY_COUNT++))
        start_container
    fi
done

# Überprüfen, ob die maximale Anzahl an Versuchen erreicht wurde
if (( RETRY_COUNT == RETRY_LIMIT )); then
    echo "Maximale Anzahl an Versuchen erreicht. Abbruch."
    exit 1
fi

echo "Fertig."