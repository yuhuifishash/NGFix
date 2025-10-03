MEX=48
M=16
efC=500
efC_AKNN=1500
./test/build_hnsw_ngfix_aknn \
--train_query_path /SSD/Text-to-Image/query.train.10M.fbin \
--train_gt_path /SSD/Text-to-Image/gt.train_10M_10M_top1000.bin \
--base_graph_path /SSD/models/NGFix/t2i10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
--metric ip_float --efC_AKNN ${efC_AKNN} \
--result_index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index \


# MEX=48
# M=16
# efC=500
# efC_AKNN=1500
# ./test/build_hnsw_ngfix_aknn \
# --train_query_path /SSD/MainSearch/train.1M.fbin \
# --train_gt_path /SSD/MainSearch/gt.train.1M.bin \
# --base_graph_path /SSD/models/NGFix/mainse_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --metric ip_float --efC_AKNN ${efC_AKNN} \
# --result_index_path /SSD/models/NGFix/mainse_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index \

