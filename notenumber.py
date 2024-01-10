# 計 算 機 科 学 実 験 及 演 習 4「 音 響 信 号 処 理 」
# サ ン プ ル ソ ー ス コ ー ド
#
# ノ ー ト ナ ン バ ー と 周 波 数 の 変 換

import math

# ノ ー ト ナ ン バ ー か ら 周 波 数 へ
def nn2hz(notenum):
    return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周 波 数 か ら ノ ー ト ナ ン バ ー へ
def hz2nn(frequency):
    return int(round(12.0 * (math.log(frequency / 440.0) / math.log(2.0)))) + 69
