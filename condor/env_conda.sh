# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/afs/cern.ch/work/z/zhibin/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/afs/cern.ch/work/z/zhibin/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/afs/cern.ch/work/z/zhibin/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate snd-pytorch
# <<< conda initialize <<<
