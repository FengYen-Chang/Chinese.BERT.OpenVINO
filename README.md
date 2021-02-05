# Chinese.BERT.OpenVINO
This repository shows the question and answering demo with distilled Chinese BERT model by OpenVINO. 

-------

First, I would like to thanks the author [xiongma](https://github.com/xiongma) who shared the roberta wwm base distilled model on his [repository](https://github.com/xiongma/roberta-wwm-base-distill) and it is trained by Chinese dataset. And we based on this model to run SQUAD task on OpenVINO. Below are whole steps that I do to enable SQUAD task and run it with OpenVINO. 

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

	> If you see the OOM issue, please reduce the batch size or max sequence length. In my case, I am using GTX1080Ti to do the fine tuning, and I set the training batch size and the max sequence length as `32` and  `256`, respectively.

### Evaluate the fine tuning result

After the fine tuning completed, you will see the predict result, `dev_predictions.json`, which is svaed at the `${OUTPUT_DIR}`. And we can use `cmrc2018_evaluate.py` to evaluate the result.
> Please use [this](https://github.com/FengYen-Chang/cmrc2018/blob/master/baseline/cmrc2018_evaluate.py), `cmrc2018_evaluate.py`, to do the evaluation if you are using Python3.
	
* Run evaluate

	```sh
	python cmrc2018_evaluate.py ${DATA_DIR}/cmrc2018_dev.json ${OUTPUT_DIR}/predictions.json
	```
	> If you meet below error, please run $python and follow the command to download the `punkt` sentence tokenizer.
	> ```sh
	> Resource punkt not found.
	> Please use the NLTK Downloader to obtain the resource:
	> 
  	> >>> import nltk
  	> >>> nltk.download('punkt')
	> ```
  	> For more information see: https://www.nltk.org/data.html
	> 
	
* Fine tuning result on SQUAD task:
	```sh
	{"AVERAGE": "66.298", "F1": "76.616", "EM": "55.980", "TOTAL": 3219, "SKIP": 0, "FILE": "dev_predictions.json"}
	```
	> This result just for reference, your result might not same as me.
	
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
### Convert the frozen tensorflow model to Intermediate Representation (IR)

Please use below command to run the Model Optimizer to get the IR model

```sh
export OUTPUT_DIR=/path/to/save/result/and/tuned_model

python mo.py --input_model=${OUTPUT_DIR}/inference_graph.pb \
	-o ${OUTPUT_DIR} \
  	--input "IteratorGetNext:0{i32}[1 256],IteratorGetNext:1{i32}[1 256],IteratorGetNext:3{i32}[1 256]" \
  	--disable_nhwc_to_nchw
```

### Running the demo `squad_openvino.py` with OpenVINO

Running the application with the `-h` option yields the following usage message:

```sh
python squad_openvino.py -h
usage: squad_openvino.py [-h] [-d DEVICE] -v VOCAB -m MODEL -i INPUT_DATA
                         [--max-seq-length MAX_SEQ_LENGTH]
                         [--doc-stride DOC_STRIDE]
                         [--max-query-length MAX_QUERY_LENGTH]
                         [-qn QUESTION_NUMBER] [-mal MAX_ANSWER_LENGTH]
                         [-nbest NUM_OF_BEST_SET]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
  -v VOCAB, --vocab VOCAB
  -m MODEL, --model MODEL
  -i INPUT_DATA, --input-data INPUT_DATA
  --max-seq-length MAX_SEQ_LENGTH
  --doc-stride DOC_STRIDE
  --max-query-length MAX_QUERY_LENGTH
  -qn QUESTION_NUMBER, --question_number QUESTION_NUMBER
  -mal MAX_ANSWER_LENGTH, --max_answer_length MAX_ANSWER_LENGTH
  -nbest NUM_OF_BEST_SET, --num_of_best_set NUM_OF_BEST_SET
```

#### Running Inference

* Input file `cmrc2018_patch.json`:
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
	python squad_openvino.py \
		-v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
		-m ${MODEL_DIR}/inference_graph.xml \
		-i ${INPUT_FILE_DIR}/cmrc2018_patch.json \
		-nbest 20 \ 
		-qn 0
	```

* Output:
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
	> For the another results, please check [this](https://github.com/FengYen-Chang/Chinese.BERT.OpenVINO/blob/main/performance.md) page.

#### Traditional Chinese Support

As the label file is **Simplify Chinese**, please convert use [chinese-converter](https://pypi.org/project/chinese-converter/) to convert it to Traditional Chinese and then run the inference. The result will like below.

```sh
[INFO] 2021-02-05 17:27:26,527 Initializing Inference Engine
[INFO] 2021-02-05 17:27:26,533 Plugin version is 2.1.2021.2.0-1877-176bdf51370-releases/2021/2
[INFO] 2021-02-05 17:27:26,533 Loading network files:
        ../frozen_model//inference_graph.xml
        ../frozen_model//inference_graph.bin
[INFO] 2021-02-05 17:27:26,662 Loading model to the CPU
[INFO] 2021-02-05 17:27:27,114 Inputs number: 3
        - IteratorGetNext/placeholder_out_port_0 : [1, 256]
        - IteratorGetNext/placeholder_out_port_1 : [1, 256]
        - IteratorGetNext/placeholder_out_port_3 : [1, 256]
[INFO] 2021-02-05 17:27:27,114 Outputs number: 2
        - unstack/Squeeze_ : [1, 256]
        - unstack/Squeeze_527 : [1, 256]
**********read_squad_examples complete!**********
[INFO] 2021-02-05 17:27:27,131 Load 4 examples
Content:  基於《跑跑卡丁車》與《泡泡堂》上所開發的遊戲，由韓國Nexon開發與發行。中國大陸由盛大遊戲運營，這是Nexon時隔6年再次授予盛大網絡其遊戲運營權。臺灣由遊戲橘子運營。玩家以水槍、小槍、錘子或是水炸彈泡封敵人(玩家或NPC)，即為一泡封，將水泡擊破為一踢爆。若水泡未在時間內踢爆，則會從水泡中釋放或被隊友救援(即為一救援)。每次泡封會減少生命數，生命數耗完即算為踢爆。重生者在一定時間內為無敵狀態，以踢爆數計分較多者獲勝，規則因模式而有差異。以2V2、4V4隨機配對的方式，玩家可依勝場數爬牌位(依序為原石、銅牌、銀牌、金牌、白金、鑽石、大師)，可選擇經典、熱血4分鐘內不得進行配對(每次中離+4分鐘)。開放時間為暑假或寒假期間內不定期開放，8人經典模式隨機配對，采計分方式，活動時間內分數越多，終了時可依該名次獲得獎勵。
Question:  生命數耗完即算為什麼？
[INFO] 2021-02-05 17:27:27,207 Inference time : 0.061 sec
[INFO] 2021-02-05 17:27:27,253 Inference time : 0.045 sec
Answer:  踢
```

### Run All cmrc2018 Dataset

```sh
python run_all_squad_openvino.py \
	-v ${CHINESE_VOCAB_FILE_DIR}/vocab.txt \
	-m ${MODEL_DIR}/inference_graph.xml \
	-i ${INPUT_FILE_DIR}/cmrc2018_dev.json \
	-nbest 20 
```

### Reference

* [bert](https://github.com/google-research/bert)
* [roberta-wwm-base-distill](https://github.com/xiongma/roberta-wwm-base-distill)
* [cmrc2018](https://github.com/ymcui/cmrc2018)
* [Convert TensorFlow* BERT Model to the Intermediate Representation](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_BERT_From_Tensorflow.html)
* [OpenVINOでfine-tuningしたBERTモデルを使用する](https://tech.gmogshd.com/fine-tuning-bert-model-with-openvino/)
