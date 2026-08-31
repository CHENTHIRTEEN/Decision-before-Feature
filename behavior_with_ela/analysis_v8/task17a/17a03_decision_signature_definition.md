# 17a03_decision_signature_definition

## 定义

动作集合为 SHADE、L-SHADE、CSO。中心化损失为每个 state 的动作损失减去三动作均值；pairwise margin 为 L_b-L_a，正值表示动作 a 的损失较低。每对 margin 除以对应两个 solver 的 fold-local practical noise scale 的较大值。

## 保存

state_decision_signatures 表保存原始损失、中心化损失、margin、normalized_margin 与 practical action set；该表不使用 raw argmin 作为主标签。
