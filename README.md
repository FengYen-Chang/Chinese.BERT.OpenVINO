# Chinese.BERT.OpenVINO
This repository shows the question and answering demo with distilled Chinese BERT model by OpenVINO. 

-------

First, I would like to thanks the author [xiongma](https://github.com/xiongma) who shared the roberta wwm base distilled model on his [repo.](https://github.com/xiongma/roberta-wwm-base-distill) and it is trained by Chinese dataset. And we based on this model to run SQUAD task on OpenVINO. Below are whole steps that I do to enable SQUAD task and run it with OpenVINO. 

-------

### Preparation

* Check this [table](https://github.com/xiongma/roberta-wwm-base-distill#model-download) and download the prefer one from BaiduYun.
	> In this repo., I am using `Roberta-wwm-ext-large-3layers-distill, Chinese`.
	
* Download CMRC2018 Dataset
	```sh
	git clone https://github.com/ymcui/cmrc2018.git
	```

### Fine tuning the BERT model for SQUAD task

After preparation, you already have the pretrained and distilled BERT model and the Chinese SQUAD dataset, `CMRC2018`. 

* Run fine tuning

	```sh
	cd cmrc2018/baseline

	export PATH_TO_BERT=/path/to/distilled/bert/model/3layers_large
	export DATA_DIR=/path/to/cmrc2018/squad-style-data
	export OUTPUT_DIR=/path/to/save/result/and/tuned_model

	python run_cmrc2018_drcd_baseline.py \
		--vocab_file=${PATH_TO_BERT}/vocab.txt \
		--bert_config_file=${PATH_TO_BERT}/bert_config.json \
		--init_checkpoint=${PATH_TO_BERT}/bert_model.ckpt \
		--do_train=True \
		--train_file=${DATA_DIR}/cmrc2018_train.json \
		--do_predict=True \
		--predict_file=${DATA_DIR}/cmrc2018_dev.json \
		--train_batch_size=32 \
		--num_train_epochs=2 \
		--max_seq_length=256 \
		--doc_stride=128 \
		--learning_rate=3e-5 \
		--save_checkpoints_steps=1000 \
		--output_dir=${OUTPUT_DIR} \
		--do_lower_case=False \
		--use_tpu=False
	```

	> If you see the OOM issue, please reduce the batch size or max sequence length. In my case, I set the training batch size and the max sequence length as `32` and  `256`, respectively.

### Evaluate the fine tuning result

After the fine tuning completed, you will see the predict result, `dev_predictions.json`, which is svaed at the `${OUTPUT_DIR}`. And we can use `cmrc2018_evaluate.py` to evaluate the result.
> Please use [this `cmrc2018_evaluate.py`](https://github.com/FengYen-Chang/cmrc2018/blob/master/baseline/cmrc2018_evaluate.py) to do the test if you are using Python3.
	
* Run evaluate

	```sh
	python cmrc2018_evaluate.py ${DATA_DIR}/cmrc2018_dev.json ${OUTPUT_DIR}/predictions.json
	```
* Fine tuning result on SQUAD task:
	```sh
	{"AVERAGE": "66.298", "F1": "76.616", "EM": "55.980", "TOTAL": 3219, "SKIP": 0, "FILE": "../../cmrc_test_output_2/dev_predictions.json"}
	```
### Frozen tenserflow model

Before run the inference on OpenVINO, we need to freeze the model to `.pb` format for model Optimizer to convert it to IR after fine tuning process is done. In here, I am refering this [page](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_BERT_From_Tensorflow.html) to convert the model to IR and below are the steps which I done.

1. Open the file `modeling.py` which in the `cmrc2018/baseline/` in the text editor and common out below 2 lines. They should look like this: 
	```py
	# if not non_static_indexes:
	#   return shape
	```
	
2. Open the file run_cmrc2018_drcd_baseline.py and insert the following code after the line 667: 
	
	```py
	import os, sys
	from tensorflow.python.framework import graph_io
	with tf.Session(graph=tf.get_default_graph()) as sess:
	  (assignment_map, initialized_variable_names) = \
	    modeling.get_assignment_map_from_checkpoint(tf.trainable_variables(), init_checkpoint)
	  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
	  sess.run(tf.global_variables_initializer())
	  frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["unstack"])
	  graph_io.write_graph(frozen, FLAGS.output_dir, 'inference_graph.pb', as_text=False)
	print('BERT frozen model path {}'.format(os.path.join(os.path.dirname(__file__), 'inference_graph.pb')))
	sys.exit(0)
	```
	Lines before the inserted code should look like this: 
	
	```py
	(start_logits, end_logits) = create_model(
		bert_config=bert_config,
		is_training=is_training,
		input_ids=input_ids,
		input_mask=input_mask,
		segment_ids=segment_ids,
		input_span_mask=input_span_mask,
		use_one_hot_embeddings=use_one_hot_embeddings)
	```

3. Run `run_cmrc2018_drcd_baseline.py` to get the `.pb` file.

	```sh
	export PATH_TO_BERT=/path/to/distilled/bert/model/3layers_large
	export DATA_DIR=/path/to/cmrc2018/squad-style-data
	export OUTPUT_DIR=/path/to/save/result/and/tuned_model

	python run_cmrc2018_drcd_baseline.py \
		--vocab_file=${PATH_TO_BERT}/vocab.txt \
		--bert_config_file=${PATH_TO_BERT}/bert_config.json \
		--init_checkpoint=${OUTPUT_DIR}/model.ckpt-2059 \
		--do_train=False \
		--train_file=${DATA_DIR}/cmrc2018_train.json \
		--do_predict=True \
		--predict_file=${DATA_DIR}/cmrc2018_dev.json \
		--train_batch_size=32 \
		--num_train_epochs=2 \
		--max_seq_length=256 \
		--doc_stride=128 \
		--learning_rate=3e-5 \
		--save_checkpoints_steps=1000 \
		--output_dir=${OUTPUT_DIR} \
		--do_lower_case=False \
		--use_tpu=False
	```

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
