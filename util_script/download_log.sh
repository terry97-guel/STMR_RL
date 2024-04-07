# declare -a MOTIONS=("trot0" "trot1" "pace0" "pace1" "hopturn" "sidesteps")
declare -a MOTIONS=("sidesteps")
# declare -a MOTIONS=("pace0")
declare -a IPS=("64")
# declare -a IPS=("50")
# declare -a IPS=("64")


# for MOTION in "${MOTIONS[@]}"; do
for MOTION in "${MOTIONS[@]}"; do
    # echo "Downloading logs for ${MOTION}"
    for IP in "${IPS[@]}"; do
        echo "Downloading logs for ${MOTION} from ${IP}"
        # if IP is one of [50,63,64], then the command is:
        if [ $IP -eq 50 ] || [ $IP -eq 63 ] || [ $IP -eq 64 ]; then
            # rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.${IP}:/home/taerim/AMP-STMR/logs/STMR/${MOTION}/ logs${IP}/STMR/${MOTION}/ &
            rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.${IP}:/home/taerim/AMP-STMR/logs/RAND/${MOTION}/ logs${IP}/RAND/${MOTION}/ &
        else
            # rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.${IP}:/home/taerim/taerim/AMP-STMR/logs/STMR/${MOTION}/ logs${IP}/STMR/${MOTION}/ &
            rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.${IP}:/home/taerim/taerim/AMP-STMR/logs/RAND/${MOTION}/ logs${IP}/RAND/${MOTION}/ &
        fi
        # rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@
        # rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.${IP}:/home/taerim/AMP-STMR/logs/STMR/${MOTION}/ logs${IP}/STMR/${MOTION}/
    done
    wait
done


# rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.50:/home/taerim/AMP-STMR/logs/ logs50/ &
# rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.63:/home/taerim/AMP-STMR/logs/ logs63/&
# rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.64:/home/taerim/AMP-STMR/logs/ logs64/ &
# rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.65:/home/taerim/taerim/AMP-STMR/logs/ logs65/ &
# wait