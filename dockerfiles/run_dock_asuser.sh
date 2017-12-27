#!/bin/bash

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

container=tensorflow/tensorflow:1.3.0-devel-gpu
datamnts=''

usage() {
cat <<EOF
Usage: $0 [-h|--help]
    [--container=docker-container] [--datamnts=dir1,dir2,...]

    Sets up a docker container environment with user privileges. Use equal
    sign "=" for arguments, do not separate by space.

    --container - Docker container tag/url.
        Default: ${container}

    --datamnts - Data directory(s) to mount into the container. Comma separated.

    -h|--help - Displays this help.

EOF
}

remain_args=()

while getopts ":h-" arg; do
    case "${arg}" in
    h ) usage
        exit 2
        ;;
    - ) [ $OPTIND -ge 1 ] && optind=$(expr $OPTIND - 1 ) || optind=$OPTIND
        eval _OPTION="\$$optind"
        OPTARG=$(echo $_OPTION | cut -d'=' -f2)
        OPTION=$(echo $_OPTION | cut -d'=' -f1)
        case $OPTION in
        --container ) larguments=yes; container="$OPTARG"  ;;
        --datamnts ) larguments=yes; datamnts="$OPTARG"  ;;
        --help ) usage; exit 2 ;;
        --* ) remain_args+=($_OPTION) ;;
        esac
        OPTIND=1
        shift
        ;;
    esac
done

# grab all other remaning args.
remain_args+=($@)

# mntdata=$([[ ! -z "${datamnt// }" ]] && echo "-v ${datamnt}:${datamnt}:ro" )
mntdata=''
if [ ! -z "${datamnts// }" ]; then
    for mnt in ${datamnts//,/ } ; do
        mntdata="-v ${mnt}:${mnt} ${mntdata}"
    done
fi

# deb/ubuntu based containers
SOMECONTAINER=$container  # deb/ubuntu based containers

USEROPTS="-u $(id -u):$(id -g) -e HOME=$HOME -e USER=$USER -v $HOME:$HOME"
getent group > group
getent passwd > passwd

# append any special users from the container
nvidia-docker run --rm -ti \
  $USEROPTS \
  -w $PWD --entrypoint=bash $SOMECONTAINER -c 'cat /etc/group' >> passwd

# run as user with my privileges and group mapped into the container
# "$(hostname)_contain" \
# nvidia-docker run --rm -ti --name=mydock \
nvidia-docker run -d -t --name=mydock --net=host \
  $USEROPTS $mntdata \
  --hostname "$(hostname)_contain" \
  -v $PWD/passwd:/etc/passwd:ro -v $PWD/group:/etc/group:ro \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -w $PWD $SOMECONTAINER


# docker exec -it -u $(id -u):$(id -g) mydock bash
# docker exec -it mydock \
#     bash --init-file setup_my_keras_psgcluster_env.sh
# running some init makes a performance difference??? I don't know why.
docker exec -it mydock \
    bash --init-file ${_basedir}/blank_init.sh
# docker exec -it mydock bash

docker stop mydock && docker rm mydock
