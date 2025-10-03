MEX=48
M=16
efC=500
./test/build_hnsw_ngfix \
--train_query_path /SSD/Text-to-Image/query.train.10M.fbin \
--train_gt_path /SSD/Text-to-Image/gt.train_10M_10M_top1000.bin \
--base_graph_path /SSD/models/NGFix/t2i10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
--metric ip_float \
--result_index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \

# MEX=48
# M=16
# efC=500
# ./test/build_hnsw_ngfix \
# --train_query_path /SSD/LAION/LAION_train_query_textemb_10M.fbin \
# --train_gt_path /SSD/LAION/gt.train10M.bin \
# --base_graph_path /SSD/models/NGFix/laion10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --metric ip_float \
# --result_index_path /SSD/models/NGFix/laion10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \


# MEX=48
# M=16
# efC=500
# ./test/build_hnsw_ngfix \
# --train_query_path /SSD/MainSearch/mainse_query_train.fbin \
# --train_gt_path /SSD/MainSearch/mainse_query_train_gt.bin \
# --base_graph_path /SSD/models/NGFix/mainse_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --metric ip_float \
# --result_index_path /SSD/models/NGFix/mainse_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \

