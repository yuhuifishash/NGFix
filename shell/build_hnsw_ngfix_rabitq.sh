MEX=48
M=16
efC=500
bits=8
./test/build_hnsw_ngfix_rabitq \
--train_query_path /SSD/WebVid/webvid.query.train.2.5M.fbin \
--train_gt_path /SSD/WebVid/gt.train.top500.bin \
--base_graph_path /SSD/models/NGFix/webvid_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \
--metric ip_rabitq \
--result_index_path /SSD/models/NGFix/webvid_HNSWNGFixRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \

# MEX=48
# M=16
# efC=500
# bits=8
# ./test/build_hnsw_ngfix_rabitq \
# --train_query_path /SSD/Text-to-Image/query.train.10M.fbin \
# --train_gt_path /SSD/Text-to-Image/gt.train_10M_10M_top1000.bin \
# --base_graph_path /SSD/models/NGFix/t2i10M_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \
# --metric ip_rabitq \
# --result_index_path /SSD/models/NGFix/t2i10M_HNSWNGFixRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \

