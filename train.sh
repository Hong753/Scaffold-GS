function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

lod=0
iterations=30_000
warmup="False"
frame_selection_stride=1
frame_selection_max_size=1024
frame_selection_workers=4
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --voxel_size) vsize="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        --frame_selection_stride) frame_selection_stride="$2"; shift ;;
        --frame_selection_max_size) frame_selection_max_size="$2"; shift ;;
        --frame_selection_workers) frame_selection_workers="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$warmup" = "True" ]; then
    python train.py --eval -s ${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --frame_selection_stride "${frame_selection_stride}" --frame_selection_max_size "${frame_selection_max_size}" --frame_selection_workers "${frame_selection_workers}" --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval -s ${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --frame_selection_stride "${frame_selection_stride}" --frame_selection_max_size "${frame_selection_max_size}" --frame_selection_workers "${frame_selection_workers}" --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
fi
