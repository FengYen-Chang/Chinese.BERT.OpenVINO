# Chinese.BERT.OpenVINO
This repository shows the question and answering demo with distilled Chinese BERT model. 

### Inference result

* Input file `cmrc2018_patched.json`:
```json
{
  "version": "v1.0", 
  "data": [
    {
      "paragraphs": [
        {
          "id": "TRIAL_800", 
          "context": "基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师) ，可选择经典、热血、狙击等模式进行游戏。若游戏中离，则4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。", 
          "qas": [
            {
              "question": "生命数耗完即算为什么？", 
              "id": "TRIAL_800_QUERY_0", 
              "answers": [
                {
                  "text": "踢爆", 
                  "answer_start": 127
                }
              ]
            }
          ]
        }
      ], 
      "id": "TRIAL_800", 
      "title": "泡泡战士"
    }
  ]
}
```
* Run inference:
```sh
python squad_openvino.py -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt -m ${MODEL_DIR}/inference_graph.xml -i ${INPUT_FILE_DIR}/cmrc2018_patched.json 
```

* Output:
```sh
[INFO] 2021-01-27 14:37:31,794 Initializing Inference Engine
[INFO] 2021-01-27 14:37:31,800 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
[INFO] 2021-01-27 14:37:31,800 Loading network files:
	/home/john-tbh/Documents/bert/frozen_model/inference_graph.xml
	/home/john-tbh/Documents/bert/frozen_model/inference_graph.bin
[INFO] 2021-01-27 14:37:31,932 Loading model to the CPU
[INFO] 2021-01-27 14:37:32,424 Inputs number: 3
	- IteratorGetNext/placeholder_out_port_0 : [1, 256]
	- IteratorGetNext/placeholder_out_port_1 : [1, 256]
	- IteratorGetNext/placeholder_out_port_3 : [1, 256]
[INFO] 2021-01-27 14:37:32,424 Outputs number: 2
	- unstack/Squeeze_ : [1, 256]
	- unstack/Squeeze_527 : [1, 256]
[INFO] 2021-01-27 14:37:32,424 Start tokenization
**********read_squad_examples complete!**********
[INFO] 2021-01-27 14:37:32,436 Load 4 examples
tokens: [CLS] 生 命 数 耗 完 即 算 为 什 么 ？ [SEP] 基 于 《 跑 跑 卡 丁 车 》 与 《 泡 泡 堂 》 上 所 开 发 的 游 戏 ， 由 韩 国 [UNK] 开 发 与 发 行 。 中 国 大 陆 由 盛 大 游 戏 运 营 ， 这 是 [UNK] 时 隔 6 年 再 次 授 予 盛 大 网 络 其 游 戏 运 营 权 。 台 湾 由 游 戏 橘 子 运 营 。 玩 家 以 水 枪 、 小 枪 、 锤 子 或 是 水 炸 弹 泡 封 敌 人 ( 玩 家 或 [UNK] ) ， 即 为 一 泡 封 ， 将 水 泡 击 破 为 一 踢 爆 。 若 水 泡 未 在 时 间 内 踢 爆 ， 则 会 从 水 泡 中 释 放 或 被 队 友 救 援 ( 即 为 一 救 援 ) 。 每 次 泡 封 会 减 少 生 命 数 ， 生 命 数 耗 完 即 算 为 踢 爆 。 重 生 者 在 一 定 时 间 内 为 无 敌 状 态 ， 以 踢 爆 数 计 分 较 多 者 获 胜 ， 规 则 因 模 式 而 有 差 异 。 以 [UNK] 、 [UNK] 随 机 配 对 的 方 式 ， 玩 家 可 依 胜 场 数 爬 牌 位 ( 依 序 为 原 石 、 铜 [SEP]
[INFO] 2021-01-27 14:37:32,439 Complete tokenization
[INFO] 2021-01-27 14:37:32,439 Predict start
[INFO] 2021-01-27 14:37:32,501 Inference time : 0.062 sec
185 186
踢爆
```
