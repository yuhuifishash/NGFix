# MEX=48
# M=16
# efC=500
# taskset -c 1 search_hnsw_ngfix \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i_K10.csv \
# --index_path /SSD/models/NGFix/t2i10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \


MEX=48
M=16
efC=500
taskset -c 1 search_hnsw_ngfix \
--test_query_path /SSD/Text-to-Image/query.10k.fbin \
--test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
--metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i_K10.csv \
--index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \