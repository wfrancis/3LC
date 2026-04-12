#!/usr/bin/env bash
# monitor_training.sh - Monitor YOLOv8 training progress
# Reads the training output and shows mAP scores, progress, and ETA.

set -euo pipefail

LOG="/private/tmp/claude-501/-Users-william-3LC/8b95af9f-6f31-4d38-885f-c199d2705911/tasks/brc3bcw2x.output"

if [[ ! -f "$LOG" ]]; then
    echo "ERROR: Training log not found at $LOG"
    exit 1
fi

# Strip ANSI escape codes and split overwritten lines (\r or \e[K)
clean() {
    sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tr '\r' '\n'
}

CLEANED=$(cat "$LOG" | clean)

# --- Header ---
echo "=============================================="
echo "  YOLOv8 Training Monitor"
echo "=============================================="
echo ""

# --- Total epochs from config ---
TOTAL_EPOCHS=$(echo "$CLEANED" | grep -o 'epochs=[0-9]*' | head -1 | cut -d= -f2)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}

# --- Current epoch/batch progress ---
# Training lines look like: "  2/20  4.53G  3.841  3.887  4.109  330  640: 4% ... 21/491 2.7s/it ..."
LAST_TRAIN=$(echo "$CLEANED" | grep -E '^\s+[0-9]+/'"$TOTAL_EPOCHS"'\s' | tail -1)

if [[ -n "$LAST_TRAIN" ]]; then
    CUR_EPOCH=$(echo "$LAST_TRAIN" | awk '{print $1}' | cut -d/ -f1)
    BATCH_PROGRESS=$(echo "$LAST_TRAIN" | grep -oE '[0-9]+/[0-9]+' | tail -1)
    CUR_BATCH=$(echo "$BATCH_PROGRESS" | cut -d/ -f1)
    TOTAL_BATCHES=$(echo "$BATCH_PROGRESS" | cut -d/ -f2)
    SPEED=$(echo "$LAST_TRAIN" | grep -oE '[0-9.]+s/it' | tail -1)

    # Extract losses
    BOX_LOSS=$(echo "$LAST_TRAIN" | awk '{print $3}')
    CLS_LOSS=$(echo "$LAST_TRAIN" | awk '{print $4}')
    DFL_LOSS=$(echo "$LAST_TRAIN" | awk '{print $5}')

    echo "PROGRESS"
    echo "----------------------------------------------"
    printf "  Epoch:        %s / %s\n" "$CUR_EPOCH" "$TOTAL_EPOCHS"
    printf "  Batch:        %s / %s\n" "$CUR_BATCH" "$TOTAL_BATCHES"
    printf "  Speed:        %s\n" "${SPEED:-n/a}"
    echo ""
    echo "CURRENT LOSSES"
    echo "----------------------------------------------"
    printf "  box_loss:     %s\n" "$BOX_LOSS"
    printf "  cls_loss:     %s\n" "$CLS_LOSS"
    printf "  dfl_loss:     %s\n" "$DFL_LOSS"

    # --- ETA calculation ---
    # Remaining batches this epoch + remaining full epochs
    if [[ -n "$SPEED" && -n "$CUR_BATCH" && -n "$TOTAL_BATCHES" && -n "$CUR_EPOCH" ]]; then
        SECS_PER_IT=$(echo "$SPEED" | sed 's/s\/it//')
        REMAINING_BATCHES=$((TOTAL_BATCHES - CUR_BATCH))
        REMAINING_EPOCHS=$((TOTAL_EPOCHS - CUR_EPOCH))
        # Each epoch also has a validation pass (~31 batches at ~8s each = ~250s)
        VAL_TIME=250
        TOTAL_REMAINING_SECS=$(echo "$SECS_PER_IT $REMAINING_BATCHES $REMAINING_EPOCHS $TOTAL_BATCHES $VAL_TIME" | \
            awk '{printf "%.0f", ($1 * $2) + ($3 * $4 * $1) + (($3 + 1) * $5)}')

        HOURS=$((TOTAL_REMAINING_SECS / 3600))
        MINS=$(((TOTAL_REMAINING_SECS % 3600) / 60))
        SECS=$((TOTAL_REMAINING_SECS % 60))

        echo ""
        echo "ESTIMATED TIME REMAINING"
        echo "----------------------------------------------"
        if [[ $HOURS -gt 0 ]]; then
            printf "  ETA:          %dh %dm %ds\n" "$HOURS" "$MINS" "$SECS"
        elif [[ $MINS -gt 0 ]]; then
            printf "  ETA:          %dm %ds\n" "$MINS" "$SECS"
        else
            printf "  ETA:          %ds\n" "$SECS"
        fi
    fi
else
    echo "  (no training batches detected yet)"
fi

# --- Completed epoch mAP scores ---
echo ""
echo "VALIDATION RESULTS (mAP per epoch)"
echo "----------------------------------------------"
printf "  %-8s  %-8s  %-8s  %-10s  %-10s\n" "Epoch" "P" "R" "mAP50" "mAP50-95"
printf "  %-8s  %-8s  %-8s  %-10s  %-10s\n" "-----" "------" "------" "--------" "--------"

# Each "all" line follows a completed epoch's validation
ALL_LINES=$(echo "$CLEANED" | grep -E '^\s+all\s+[0-9]')

if [[ -z "$ALL_LINES" ]]; then
    echo "  (no completed epochs yet)"
else
    EPOCH_NUM=0
    while IFS= read -r line; do
        EPOCH_NUM=$((EPOCH_NUM + 1))
        P=$(echo "$line" | awk '{print $4}')
        R=$(echo "$line" | awk '{print $5}')
        MAP50=$(echo "$line" | awk '{print $6}')
        MAP50_95=$(echo "$line" | awk '{print $7}')
        printf "  %-8s  %-8s  %-8s  %-10s  %-10s\n" "$EPOCH_NUM" "$P" "$R" "$MAP50" "$MAP50_95"
    done <<< "$ALL_LINES"

    # Highlight best mAP50
    BEST_MAP50=$(echo "$ALL_LINES" | awk '{print $6}' | sort -rn | head -1)
    BEST_MAP50_95=$(echo "$ALL_LINES" | awk '{print $7}' | sort -rn | head -1)
    echo ""
    printf "  Best mAP50:     %s\n" "$BEST_MAP50"
    printf "  Best mAP50-95:  %s\n" "$BEST_MAP50_95"
fi

echo ""
echo "=============================================="
