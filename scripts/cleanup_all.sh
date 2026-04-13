#!/bin/bash

# ============================================================================
# 一键清理所有训练进程、端口和GPU
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🧹 PhysioTouch 环境清理工具"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# Step 1: 清理训练进程
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: 清理训练进程"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 查找所有Stage训练进程
STAGE_PROCS=$(ps aux | grep -E "Stage[12]_main.py|train_stage" | grep -v grep | wc -l)

if [ "$STAGE_PROCS" -gt 0 ]; then
    echo "📋 发现 $STAGE_PROCS 个训练进程，正在清理..."
    
    # 清理Stage1和Stage2训练进程
    pkill -9 -f "Stage1_main.py" 2>/dev/null && echo "  ✅ 清理 Stage1 进程" || echo "  ⚪ 无 Stage1 进程"
    pkill -9 -f "Stage2_main.py" 2>/dev/null && echo "  ✅ 清理 Stage2 进程" || echo "  ⚪ 无 Stage2 进程"
    
    # 清理torchrun进程
    pkill -9 -f "torchrun" 2>/dev/null && echo "  ✅ 清理 torchrun 进程" || echo "  ⚪ 无 torchrun 进程"
    
    # 清理Python训练脚本
    pkill -9 -f "train_stage" 2>/dev/null && echo "  ✅ 清理训练脚本进程" || echo "  ⚪ 无训练脚本进程"
    
    sleep 1
    echo ""
    echo "✅ 训练进程清理完成"
else
    echo "✅ 无训练进程运行"
fi

echo ""

# ============================================================================
# Step 2: 清理端口占用
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: 清理端口占用"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 清理常用的分布式训练端口
PORTS=(29500 29501 29502 29503 29504 29505)
CLEANED=0

for PORT in "${PORTS[@]}"; do
    PIDS=$(lsof -ti :$PORT 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "$PIDS" | xargs -r kill -9 2>/dev/null
        echo "  ✅ 清理端口 $PORT"
        CLEANED=$((CLEANED + 1))
    fi
done

if [ "$CLEANED" -eq 0 ]; then
    echo "✅ 所有常用端口均已释放"
else
    echo ""
    echo "✅ 已清理 $CLEANED 个端口"
fi

echo ""

# ============================================================================
# Step 3: 清理僵尸进程
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: 清理僵尸Python进程"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 查找占用大量GPU内存的Python进程
ZOMBIE_COUNT=0

for GPU in 0 1 2 3; do
    # 获取该GPU上的所有进程
    PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=$GPU 2>/dev/null)
    
    if [ -n "$PROCS" ]; then
        for PID in $PROCS; do
            # 检查进程是否存在且是Python进程
            if ps -p $PID > /dev/null 2>&1; then
                CMD=$(ps -p $PID -o comm= 2>/dev/null)
                if [[ "$CMD" == "python"* ]] || [[ "$CMD" == "torchrun" ]]; then
                    # 检查是否是孤儿进程（父进程已不存在）
                    PPID=$(ps -p $PID -o ppid= 2>/dev/null | tr -d ' ')
                    if ! ps -p $PPID > /dev/null 2>&1; then
                        echo "  🗑️  清理僵尸进程 PID=$PID (GPU $GPU)"
                        kill -9 $PID 2>/dev/null
                        ZOMBIE_COUNT=$((ZOMBIE_COUNT + 1))
                    fi
                fi
            fi
        done
    fi
done

if [ "$ZOMBIE_COUNT" -eq 0 ]; then
    echo "✅ 无僵尸进程"
else
    echo ""
    echo "✅ 已清理 $ZOMBIE_COUNT 个僵尸进程"
fi

echo ""

# ============================================================================
# Step 4: 等待GPU释放
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: 等待GPU释放"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sleep 2  # 等待系统清理

echo ""

# ============================================================================
# Step 5: 验证清理结果
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: 验证清理结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查训练进程
REMAINING_PROCS=$(ps aux | grep -E "Stage[12]_main.py|torchrun" | grep -v grep | wc -l)
if [ "$REMAINING_PROCS" -eq 0 ]; then
    echo "✅ 训练进程: 全部清理"
else
    echo "⚠️  训练进程: 还有 $REMAINING_PROCS 个残留"
fi

# 检查端口
REMAINING_PORTS=0
for PORT in "${PORTS[@]}"; do
    if lsof -i :$PORT >/dev/null 2>&1; then
        echo "⚠️  端口 $PORT: 仍被占用"
        REMAINING_PORTS=$((REMAINING_PORTS + 1))
    fi
done

if [ "$REMAINING_PORTS" -eq 0 ]; then
    echo "✅ 端口状态: 全部释放"
fi

# 检查GPU使用情况
echo ""
echo "📊 当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F, '{printf "  GPU %s: %5s MB / %s MB (利用率: %3s%%)\n", $1, $3, $4, $5}'

echo ""

# ============================================================================
# 总结
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "清理完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$REMAINING_PROCS" -eq 0 ] && [ "$REMAINING_PORTS" -eq 0 ]; then
    echo ""
    echo "🎉 环境已完全清理，可以启动新的训练！"
    echo ""
    echo "建议命令："
    echo "  tmux new -s Stage2-b"
    echo "  bash scripts/train_stage2b_mixed.sh"
    echo ""
else
    echo ""
    echo "⚠️  部分资源清理失败，请检查上方信息"
    echo ""
    if [ "$REMAINING_PROCS" -gt 0 ]; then
        echo "残留进程："
        ps aux | grep -E "Stage[12]_main.py|torchrun" | grep -v grep
    fi
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

