# Performace Result

This page will show the all performance data for the converted IR model.

### Original Result

```json
{
    "TRIAL_800_QUERY_0": "踢爆",
    "TRIAL_800_QUERY_1": "4分钟内不得进行配对(每次中离+4分钟",
    "TRIAL_800_QUERY_2": "水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)",
    "TRIAL_800_QUERY_3": "可选择经典、热血、狙击等模式进行游戏。"
}
```

### OpenVINO Output

* Question 0:

  Command:
  
  * Run with CPU
    ```sh
    python squad_openvino.py \
      -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
      -m ${MODEL_DIR}/inference_graph.xml \
      -i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
      -nbest 20 \ 
      -qn 0
    ```
  
  * Run with GPU
    ```sh
    python squad_openvino.py \
      -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
      -m ${MODEL_DIR}/inference_graph.xml \
      -i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
      -nbest 20 \ 
      -qn 0 \
      -d GPU
    ```
  
  Output:

  * FP32 (Run on CPU)
    ```sh
    [INFO] 2021-01-29 17:49:14,843 Initializing Inference Engine
    [INFO] 2021-01-29 17:49:14,849 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
    [INFO] 2021-01-29 17:49:14,849 Loading network files:
            /home/john-tbh/Documents/bert/frozen_model/inference_graph.xml
            /home/john-tbh/Documents/bert/frozen_model/inference_graph.bin
    [INFO] 2021-01-29 17:49:14,972 Loading model to the CPU
    [INFO] 2021-01-29 17:49:15,414 Inputs number: 3
            - IteratorGetNext/placeholder_out_port_0 : [1, 256]
            - IteratorGetNext/placeholder_out_port_1 : [1, 256]
            - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-01-29 17:49:15,414 Outputs number: 2
            - unstack/Squeeze_ : [1, 256]
            - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-01-29 17:49:15,432 Load 4 examples
    Content:  基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师)，可选择经典、热血4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。
    Question:  生命数耗完即算为什么？
    [INFO] 2021-01-29 17:49:15,506 Inference time : 0.06 sec
    [INFO] 2021-01-29 17:49:15,552 Inference time : 0.045 sec
    Answer:  踢爆
    ```
   
  * FP16 (Run on GPU)
    ```sh
    [INFO] 2021-01-29 18:22:08,599 Initializing Inference Engine
    [INFO] 2021-01-29 18:22:08,605 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
    [INFO] 2021-01-29 18:22:08,605 Loading network files:
            /home/john-tbh/Documents/bert/frozen_model/FP16/inference_graph.xml
            /home/john-tbh/Documents/bert/frozen_model/FP16/inference_graph.bin
    [INFO] 2021-01-29 18:22:08,680 Loading model to the GPU
    [INFO] 2021-01-29 18:22:09,559 Inputs number: 3
            - IteratorGetNext/placeholder_out_port_0 : [1, 256]
            - IteratorGetNext/placeholder_out_port_1 : [1, 256]
            - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-01-29 18:22:09,559 Outputs number: 2
            - unstack/Squeeze_ : [1, 256]
            - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-01-29 18:22:09,577 Load 4 examples
    Content:  基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师)，可选择经典、热血4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。
    Question:  生命数耗完即算为什么？
    [INFO] 2021-01-29 18:32:11,339 Inference time : 0.042 sec
    [INFO] 2021-01-29 18:32:11,381 Inference time : 0.041 sec
    Answer:  踢爆 
    ```

* Question 1

  Command:
  
  * Run with CPU
    ```sh
    python squad_openvino.py \
      -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
      -m ${MODEL_DIR}/inference_graph.xml \
      -i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
      -nbest 20 \ 
      -qn 1
    ```
  
  * Run with GPU
    ```sh
    python squad_openvino.py \
      -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
      -m ${MODEL_DIR}/inference_graph.xml \
      -i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
      -nbest 20 \ 
      -qn 1 \
      -d GPU
    ```
  
  Output:

  * FP32 (Run on CPU)
  
    ```sh
    [INFO] 2021-01-29 17:49:14,843 Initializing Inference Engine
    [INFO] 2021-01-29 17:49:14,849 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
    [INFO] 2021-01-29 17:49:14,849 Loading network files:
            /home/john-tbh/Documents/bert/frozen_model/inference_graph.xml
            /home/john-tbh/Documents/bert/frozen_model/inference_graph.bin
    [INFO] 2021-01-29 17:49:14,972 Loading model to the CPU
    [INFO] 2021-01-29 17:49:15,414 Inputs number: 3
            - IteratorGetNext/placeholder_out_port_0 : [1, 256]
            - IteratorGetNext/placeholder_out_port_1 : [1, 256]
            - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-01-29 17:49:15,414 Outputs number: 2
            - unstack/Squeeze_ : [1, 256]
            - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-01-29 18:00:56,802 Load 4 examples
    Content:  基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师)，可选择经典、热血4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。
    Question:  生命数耗完即算为什么？
    [INFO] 2021-01-29 18:00:56,876 Inference time : 0.06 sec
    [INFO] 2021-01-29 18:00:56,922 Inference time : 0.045 sec
    [INFO] 2021-01-29 18:00:56,969 Inference time : 0.045 sec
    Answer:  4分钟内不得进行配对(每次中离+4分钟
    ```
    
  * FP16 (Run on GPU)
    ```sh
    [INFO] 2021-01-29 18:22:38,511 Initializing Inference Engine
    [INFO] 2021-01-29 18:22:38,517 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
    [INFO] 2021-01-29 18:22:38,517 Loading network files:
            /home/john-tbh/Documents/bert/frozen_model/FP16/inference_graph.xml
            /home/john-tbh/Documents/bert/frozen_model/FP16/inference_graph.bin
    [INFO] 2021-01-29 18:22:38,592 Loading model to the GPU
    [INFO] 2021-01-29 18:22:39,476 Inputs number: 3
            - IteratorGetNext/placeholder_out_port_0 : [1, 256]
            - IteratorGetNext/placeholder_out_port_1 : [1, 256]
            - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-01-29 18:22:39,476 Outputs number: 2
            - unstack/Squeeze_ : [1, 256]
            - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-01-29 18:22:39,494 Load 4 examples
    Content:  基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师)，可选择经典、热血4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。
    Question:  生命数耗完即算为什么？
    [INFO] 2021-01-29 18:31:42,949 Inference time : 0.042 sec
    [INFO] 2021-01-29 18:31:42,990 Inference time : 0.041 sec
    [INFO] 2021-01-29 18:31:43,032 Inference time : 0.041 sec
    Answer:  4分钟内不得进行配对(每次中离+4分钟
    ```
  
* Question 2:

  Command:
  
  * Run with CPU
    ```sh
    python squad_openvino.py \
      -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
      -m ${MODEL_DIR}/inference_graph.xml \
      -i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
      -nbest 20 \ 
      -qn 2
    ```
  
  * Run with GPU
    ```sh
    python squad_openvino.py \
      -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
      -m ${MODEL_DIR}/inference_graph.xml \
      -i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
      -nbest 20 \ 
      -qn 2 \
      -d GPU
    ```
  
  Output:
  
  * FP32 (Run on CPU)

    ```sh
    [INFO] 2021-01-29 17:49:14,843 Initializing Inference Engine
    [INFO] 2021-01-29 17:49:14,849 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
    [INFO] 2021-01-29 17:49:14,849 Loading network files:
            /home/john-tbh/Documents/bert/frozen_model/inference_graph.xml
            /home/john-tbh/Documents/bert/frozen_model/inference_graph.bin
    [INFO] 2021-01-29 17:49:14,972 Loading model to the CPU
    [INFO] 2021-01-29 17:49:15,414 Inputs number: 3
            - IteratorGetNext/placeholder_out_port_0 : [1, 256]
            - IteratorGetNext/placeholder_out_port_1 : [1, 256]
            - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-01-29 17:49:15,414 Outputs number: 2
            - unstack/Squeeze_ : [1, 256]
            - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-01-29 18:01:38,108 Load 4 examples
    Content:  基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师)，可选择经典、热血4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。
    Question:  生命数耗完即算为什么？
    [INFO] 2021-01-29 18:01:38,183 Inference time : 0.06 sec
    [INFO] 2021-01-29 18:01:38,229 Inference time : 0.045 sec
    Answer:  水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)
    ```
    
  * FP16 (Run on GPU)
   
    ```sh
    [INFO] 2021-01-29 18:24:12,155 Initializing Inference Engine
    [INFO] 2021-01-29 18:24:12,161 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
    [INFO] 2021-01-29 18:24:12,161 Loading network files:
            /home/john-tbh/Documents/bert/frozen_model/FP16/inference_graph.xml
            /home/john-tbh/Documents/bert/frozen_model/FP16/inference_graph.bin
    [INFO] 2021-01-29 18:24:12,236 Loading model to the GPU
    [INFO] 2021-01-29 18:24:13,113 Inputs number: 3
            - IteratorGetNext/placeholder_out_port_0 : [1, 256]
            - IteratorGetNext/placeholder_out_port_1 : [1, 256]
            - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-01-29 18:24:13,114 Outputs number: 2
            - unstack/Squeeze_ : [1, 256]
            - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-01-29 18:24:13,131 Load 4 examples
    Content:  基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师)，可选择经典、热血4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。
    Question:  生命数耗完即算为什么？
    [INFO] 2021-01-29 18:29:17,297 Inference time : 0.042 sec
    [INFO] 2021-01-29 18:29:17,339 Inference time : 0.041 sec
    Answer:  水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)
    ```

* Question 3:

  Command:
  
  * Run with CPU
    ```sh
    python squad_openvino.py \
      -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
      -m ${MODEL_DIR}/inference_graph.xml \
      -i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
      -nbest 20 \ 
      -qn 3
    ```
  
  * Run with GPU
    ```sh
    python squad_openvino.py \
      -v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
      -m ${MODEL_DIR}/inference_graph.xml \
      -i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
      -nbest 20 \ 
      -qn 3 \
      -d GPU
    ```
  
  Output:
  
  * FP32 (Run on CPU)

    ```sh
    [INFO] 2021-01-29 17:49:14,843 Initializing Inference Engine
    [INFO] 2021-01-29 17:49:14,849 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
    [INFO] 2021-01-29 17:49:14,849 Loading network files:
            /home/john-tbh/Documents/bert/frozen_model/inference_graph.xml
            /home/john-tbh/Documents/bert/frozen_model/inference_graph.bin
    [INFO] 2021-01-29 17:49:14,972 Loading model to the CPU
    [INFO] 2021-01-29 17:49:15,414 Inputs number: 3
            - IteratorGetNext/placeholder_out_port_0 : [1, 256]
            - IteratorGetNext/placeholder_out_port_1 : [1, 256]
            - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-01-29 17:49:15,414 Outputs number: 2
            - unstack/Squeeze_ : [1, 256]
            - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-01-29 18:02:04,034 Load 4 examples
    Content:  基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师)，可选择经典、热血4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。
    Question:  生命数耗完即算为什么？
    [INFO] 2021-01-29 18:02:04,108 Inference time : 0.06 sec
    [INFO] 2021-01-29 18:02:04,154 Inference time : 0.045 sec
    Answer:  可选择经典、热血、狙击等模式进行游戏。
    ```
    
  * FP16 (Run on GPU)
    
    ```sh
    [INFO] 2021-01-29 18:25:14,715 Initializing Inference Engine
    [INFO] 2021-01-29 18:25:14,721 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
    [INFO] 2021-01-29 18:25:14,721 Loading network files:
            /home/john-tbh/Documents/bert/frozen_model/FP16/inference_graph.xml
            /home/john-tbh/Documents/bert/frozen_model/FP16/inference_graph.bin
    [INFO] 2021-01-29 18:25:14,795 Loading model to the CPU
    [INFO] 2021-01-29 18:25:15,673 Inputs number: 3
            - IteratorGetNext/placeholder_out_port_0 : [1, 256]
            - IteratorGetNext/placeholder_out_port_1 : [1, 256]
            - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-01-29 18:25:15,673 Outputs number: 2
            - unstack/Squeeze_ : [1, 256]
            - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-01-29 18:25:15,691 Load 4 examples
    Content:  基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师)，可选择经典、热血4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。
    Question:  生命数耗完即算为什么？
    [INFO] 2021-01-29 18:25:15,765 Inference time : 0.043 sec
    [INFO] 2021-01-29 18:25:15,811 Inference time : 0.041 sec
    Answer:  可选择经典、热血、狙击等模式进行游戏。
    ```

### Performance Result by Benchmark_app 

| data type | CPU | GPU | 
| :------ | :-----: | :-----: | 
| FP32 | 18.37 | 15.94 | 
| FP16 | 19.20 | 25.44 | 

> UNIT: FPS

