# LINE in TensorFlow

TensorFlow implementation of paper _LINE: Large-scale Information Network Embedding_ by Jian Tang, et al.

## Input:
* link: the the edges of graph. Each row representation an edge with format: _source_nid target_nid weight_
* label: the node labels for evaluation. Each row represents an node's label with format: _nid lid_

### Notice:
* The node id is unique for each node and is in range: [0, N) where N is the number of nodes.
* For undirected graph, need to add reversed edges in link file.
* The label id is unique for each node and is in range: [0, M) where M is the number of labels.
* Only for single label classification task.
* You can simply omit the `label` param if you donot want to evaluate the learned embeddings.

## Usage:
python line.py --link link_path --label label_path --save save_dir --embedding_size 128 --learning_rate 0.025 --num_batches 100000 --batch_size 512 --negative 5

## References
- https://github.com/snowkylin/line
- Tang, Jian, et al. "[Line: Large-scale information network embedding.](https://dl.acm.org/citation.cfm?id=2741093)" _Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee_, 2015.
