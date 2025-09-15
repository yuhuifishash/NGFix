# MEX=48
# M=16
# efC=500
# taskset -c 1 test/search_hnsw_ngfix \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/t2i10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \

MEX=48
M=16
efC=500
taskset -c 1 test/search_hnsw_ngfix \
--test_query_path /SSD/SIFT1M/sift_query.fbin \
--test_gt_path /SSD/SIFT1M/gt.query.top100.bin \
--metric l2_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i.csv \
--index_path /SSD/models/NGFix/sift1M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \



# MEX=48
# M=16
# efC=500
# taskset -c 1 test/search_hnsw_ngfix \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \


# MEX=48
# M=16
# efC=500
# taskset -c 1 test/search_hnsw_ngfix \
# --test_query_path /SSD/MainSearch/query_test_unique.fbin \
# --test_gt_path /SSD/MainSearch/gt.test_unique.bin \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_mainse.csv \
# --index_path /SSD/models/NGFix/mainse_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \