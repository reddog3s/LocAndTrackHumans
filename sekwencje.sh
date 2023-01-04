#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
read -p "Username:" username
read -p "Password:" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=rich&sfile=test/Gym_010_dips1.tar.gz&resume=1' -O 'Gym_010_dips1.tar.gz' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=rich&sfile=test/LectureHall_009_021_reparingprojector1.tar.gz&resume=1' -O 'LectureHall_009_021_reparingprojector1.tar.gz' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=rich&sfile=test/ParkingLot2_017_burpeejump1.tar.gz&resume=1' -O 'ParkingLot2_017_burpeejump1.tar.gz' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=rich&sfile=train/ParkingLot1_004_005_greetingchattingeating1.tar.gz&resume=1' -O 'ParkingLot1_004_005_greetingchattingeating1.tar.gz' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=rich&sfile=train/Pavallion_003_018_tossball.tar.gz&resume=1' -O 'Pavallion_003_018_tossball.tar.gz' --no-check-certificate --continue
