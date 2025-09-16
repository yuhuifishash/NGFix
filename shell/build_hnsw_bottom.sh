MEX=48
M=16
efC=500
./build_hnsw_bottom --base_data_path /SSD/Text-to-Image/base.10M.fbin \
--metric ip_float \
--result_hnsw_index_path /SSD/models/NGFix/t2i10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
--M ${M} --MEX ${MEX} --efC ${efC}


# MEX=48
# M=16
# efC=500
# ./test/build_hnsw_bottom --base_data_path /SSD/MainSearch/base.fbin \
# --metric ip_float \
# --result_hnsw_index_path /SSD/models/NGFix/mainse_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --M ${M} --MEX ${MEX} --efC ${efC}

# MEX=48
# M=16
# efC=500
# ./test/build_hnsw_bottom --base_data_path /SSD/SIFT1M/sift_base.fbin \
# --metric l2_float \
# --result_hnsw_index_path /SSD/models/NGFix/sift1M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --M ${M} --MEX ${MEX} --efC ${efC}