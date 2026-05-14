#!/bin/bash
WORLD="bfmc_official"
MODELS=~/BFMC/bfmc_ws/SimulatorROS2/src/models_pkg
WAIT=5

spawn() {
    local sdf=$1 name=$2
    echo ">>> SPAWN: $name"
    gz service -s /world/$WORLD/create \
      --reqtype gz.msgs.EntityFactory \
      --reptype gz.msgs.Boolean \
      --timeout 2000 \
      --req "sdf_filename: \"$sdf\", name: \"$name\", pose: {position: {x: 2.5, y: -2.07, z: 0.1}}"
}

# Spawn con rotazione 180° sull'asse Z (quaternione: z=1, w=0)
spawn_flipped() {
    local sdf=$1 name=$2
    echo ">>> SPAWN (flipped): $name"
    gz service -s /world/$WORLD/create \
      --reqtype gz.msgs.EntityFactory \
      --reptype gz.msgs.Boolean \
      --timeout 2000 \
      --req "sdf_filename: \"$sdf\", name: \"$name\", pose: {position: {x: 2.5, y: -2.07, z: 0.1}, orientation: {x: 0.0, y: 0.0, z: 1.0, w: 0.0}}"
}

remove() {
    local name=$1
    echo "<<< REMOVE: $name"
    gz service -s /world/$WORLD/remove \
      --reqtype gz.msgs.Entity \
      --reptype gz.msgs.Boolean \
      --timeout 2000 \
      --req "name: \"$name\", type: 2"
    sleep 2
}

# Pulizia iniziale
for name in test_stop test_priority test_crosswalk test_roundabout \
            test_parking test_oneway test_enter_hway test_leave_hway \
            test_no_entry; do
    gz service -s /world/$WORLD/remove \
      --reqtype gz.msgs.Entity \
      --reptype gz.msgs.Boolean \
      --timeout 1000 \
      --req "name: \"$name\", type: 2" > /dev/null 2>&1
done

echo "=============================="
echo " BFMC Traffic Sign Test"
echo " Ogni cartello rimane $WAIT secondi"
echo "=============================="
sleep 1

spawn         "$MODELS/stop_sign/model.sdf"          "test_stop"
sleep $WAIT
remove        "test_stop"

spawn         "$MODELS/priority_sign/model.sdf"      "test_priority"
sleep $WAIT
remove        "test_priority"

spawn         "$MODELS/crosswalk_sign/model.sdf"     "test_crosswalk"
sleep $WAIT
remove        "test_crosswalk"

spawn         "$MODELS/roundabout_sign/model.sdf"    "test_roundabout"
sleep $WAIT
remove        "test_roundabout"

spawn         "$MODELS/parking_sign/model.sdf"       "test_parking"
sleep $WAIT
remove        "test_parking"

spawn         "$MODELS/oneway_sign/model.sdf"        "test_oneway"
sleep $WAIT
remove        "test_oneway"

spawn_flipped "$MODELS/enter_highway_sign/model.sdf" "test_enter_hway"
sleep $WAIT
remove        "test_enter_hway"

spawn_flipped "$MODELS/leave_highway_sign/model.sdf" "test_leave_hway"
sleep $WAIT
remove        "test_leave_hway"

spawn         "$MODELS/prohibited_sign/model.sdf"    "test_no_entry"
sleep $WAIT
remove        "test_no_entry"

echo "=============================="
echo " Test completato!"
echo "=============================="
