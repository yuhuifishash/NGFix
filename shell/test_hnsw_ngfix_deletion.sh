# MEX=48
# M=16
# efC=500

# # first need to build a HNSW-NGFix*
# # delete 2M base data (we use gt.10K_8M to test)
# ./test/test_hnsw_ngfix_deletion \
# --metric ip_float \
# --result_index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_Delete2M.index \
# --index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \


# taskset -c 1 test/search_hnsw_ngfix \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_8M.bin \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_Delete2M.index \



# test deletion on HNSW_BOTTOM without NGFix*
MEX=48
M=16
efC=500
./test/build_hnsw_bottom --base_data_path /SSD/DEEP10M/base.fbin \
--metric ip_float \
--result_hnsw_index_path /SSD/models/NGFix/deep10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
--M ${M} --MEX ${MEX} --efC ${efC}


# delete 2M base data (we use gt.10K_8M to test)
./test/test_hnsw_ngfix_deletion \
--metric ip_float \
--result_index_path /SSD/models/NGFix/deep10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}_Delete2M.index \
--index_path /SSD/models/NGFix/deep10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \


taskset -c 1 test/search_hnsw_ngfix \
--test_query_path /SSD/DEEP10M/query.fbin \
--test_gt_path /SSD/DEEP10M/gt.query.8M.top100.bin \
--metric ip_float --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
--index_path /SSD/models/NGFix/deep10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}_Delete2M.index \

