# MEX=48
# M=16
# efC=500
# bits=8
# ./test/build_hnsw_bottom_rabitq --base_data_path /SSD/WebVid/webvid.base.2.5M.fbin \
# --metric ip_rabitq \
# --result_hnsw_index_path /SSD/models/NGFix/webvid_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \
# --M ${M} --MEX ${MEX} --efC ${efC} --b_bits ${bits}

# MEX=48
# M=16
# efC=500
# bits=8
# ./test/build_hnsw_bottom_rabitq --base_data_path /SSD/DEEP10M/base.fbin \
# --metric ip_rabitq \
# --result_hnsw_index_path /SSD/models/NGFix/deep10m_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \
# --M ${M} --MEX ${MEX} --efC ${efC} --b_bits ${bits}

MEX=48
M=16
efC=500
bits=8
./test/build_hnsw_bottom_rabitq --base_data_path /SSD/Text-to-Image/base.10M.fbin \
--metric ip_rabitq \
--result_hnsw_index_path /SSD/models/NGFix/t2i10M_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \
--M ${M} --MEX ${MEX} --efC ${efC} --b_bits ${bits}

# MEX=48
# M=16
# efC=500
# bits=8
# ./test/build_hnsw_bottom_rabitq --base_data_path /SSD/SIFT1M/sift_base.fbin \
# --metric l2_rabitq \
# --result_hnsw_index_path /SSD/models/NGFix/sift1M_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \
# --M ${M} --MEX ${MEX} --efC ${efC} --b_bits ${bits}


# MEX=48
# M=16
# efC=500
# bits=8
# ./test/build_hnsw_bottom_rabitq --base_data_path /SSD/DEEP10M/base.1M.fbin \
# --metric ip_rabitq \
# --result_hnsw_index_path /SSD/models/NGFix/DEEP1M_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \
# --M ${M} --MEX ${MEX} --efC ${efC} --b_bits ${bits}