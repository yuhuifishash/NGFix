MEX=48
M=16
efC=500
./build_hnsw_bottom --base_data_path /SSD/Text-to-Image/base.10M.fbin \
--metric ip_float --train_number ${train_number} \
--result_hnsw_index_path /SSD/models/NGFix/t2i10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
--M ${M} --MEX ${MEX} --efC ${efC}