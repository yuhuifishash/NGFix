MEX=48
M=16
efC=500
# delete 2M base data (we use gt.10K_8M to test)
./test_hnsw_ngfix_deletion \
--test_query_path /SSD/Text-to-Image/query.10k.fbin \
--test_gt_path /SSD/Text-to-Image/gt.10K_8M.bin \
--metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i_K10.csv \
--result_index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_Delete2M.index \
--index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \


taskset -c 1 search_hnsw_ngfix \
--test_query_path /SSD/Text-to-Image/query.10k.fbin \
--test_gt_path /SSD/Text-to-Image/gt.10K_8M.bin \
--metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i.csv \
--index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_Delete2M.index \

