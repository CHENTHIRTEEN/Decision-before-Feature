# Individual primitive definitions

正式 primitive 为 rank、movement、direction、progress、stagnation、elite distance；primary window 为 500FE，敏感性窗口仅为 200/1000FE 的协议记录。rank 与 elite distance 使用当前 population；movement 使用实际相邻 native updates 的位移并以 domain diameter 归一化；direction 使用非零 displacement 单位向量的均值范数；progress 使用当前 population IQR，并用预先指定的 1e-3 × max(1, |fitness median|) 作为退化分布的相对尺度下界；stagnation 是 fitness meaningful improvement 的 FE age；全部不使用真实最优值。
