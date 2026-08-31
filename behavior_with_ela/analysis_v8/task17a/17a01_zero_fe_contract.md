# 17a01_zero_fe_contract

## 协议

本轮只读取既有 natural 与 post-handoff 状态、三动作 1000-FE 结果和已标定噪声尺度；新增 objective FE = 0。未调用 optimizer、benchmark、ELA、selector 或闭环控制。

## 范围

自然域 1890 states；交接后域 3780 states；所有随机数由显式 SeedSequence 产生。
