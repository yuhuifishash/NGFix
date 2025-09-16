# use Text-to-Image 8M to build index first 
MEX=48
M=16
efC=500
./test/build_hnsw_bottom --base_data_path /SSD/Text-to-Image/base.8M.fbin \
--metric ip_float --train_number ${train_number} \
--result_hnsw_index_path /SSD/models/NGFix/t2i8M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
--M ${M} --MEX ${MEX} --efC ${efC}

./test/build_hnsw_ngfix \
--train_query_path /SSD/Text-to-Image/query.train.10M.fbin \
--train_gt_path /SSD/Text-to-Image/gt.10M_8M_top500.bin \
--base_graph_path /SSD/models/NGFix/t2i8M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
--metric ip_float \
--result_index_path /SSD/models/NGFix/t2i8M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \







# now we insert 2M data
./test/test_hnsw_ngfix_insertion --base_data_path /SSD/Text-to-Image/base.10M.fbin \
--train_query_path /SSD/Text-to-Image/query.train.10M.fbin \
--train_gt_path /SSD/Text-to-Image/gt.train_10M_10M_top1000.bin \
--raw_index_path /SSD/models/NGFix/t2i8M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \
--metric ip_float \
--result_index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_Insert2M.index \
--efC ${efC} --insert_st_id 8000000\

taskset -c 1 test/search_hnsw_ngfix \
--test_query_path /SSD/Text-to-Image/query.10k.fbin \
--test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
--metric ip_float --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
--index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_Insert2M.index \




