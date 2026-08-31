# 16a04 Perturbation semantics

q=0.25，sigma=0.05，unit-cube reflect，无条件部分替换，排除当前 population best。Targeted 按停滞年龄降序、近期进展升序、agent_id 排序；Random 使用相同 k 与 kernel。Perturb 评价计入 1000 FE；SHADE/L-SHADE 记忆不接收扰动成功记录，CSO 仅将被替换个体速度置零。
