MEX=48
M=16
efC=500
bits=8
taskset -c 1 test/search_hnsw_ngfix_rabitq \
--is_rerank 1 --raw_vector_path /SSD/Text-to-Image/base.10M.fbin --q_bits 8 \
--test_query_path /SSD/Text-to-Image/query.10k.fbin \
--test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
--metric ip_rabitq --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
--index_path /SSD/models/NGFix/t2i10M_HNSWNGFixRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \

# MEX=48
# M=16
# efC=500
# bits=8
# taskset -c 1 test/search_hnsw_ngfix_rabitq \
# --is_rerank 1 --raw_vector_path /SSD/DEEP10M/base.fbin --q_bits 8 \
# --test_query_path /SSD/DEEP10M/query.fbin \
# --test_gt_path /SSD/DEEP10M/gt.query.top100.bin \
# --metric ip_rabitq --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/deep10m_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \


# MEX=48
# M=16
# efC=500
# bits=8
# taskset -c 1 test/search_hnsw_ngfix_rabitq \
# --is_rerank 1 --raw_vector_path /SSD/WebVid/webvid.base.2.5M.fbin --q_bits 8 \
# --test_query_path /SSD/WebVid/webvid.query.10k.fbin \
# --test_gt_path /SSD/WebVid/gt.query.top100.bin \
# --metric ip_rabitq --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/webvid_HNSWNGFixRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \

# MEX=48
# M=16
# efC=500
# bits=8
# taskset -c 1 test/search_hnsw_ngfix_rabitq \
# --is_rerank 1 --raw_vector_path /SSD/Text-to-Image/base.10M.fbin --q_bits 32 \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --metric ip_rabitq --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/t2i10M_HNSWNGFixRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \

# MEX=48
# M=16
# efC=500
# bits=8
# taskset -c 1 test/search_hnsw_ngfix_rabitq \
# --is_rerank 1 --raw_vector_path /SSD/SIFT1M/sift_base.fbin --q_bits 8 \
# --test_query_path /SSD/SIFT1M/sift_query.fbin \
# --test_gt_path /SSD/SIFT1M/gt.query.top100.bin \
# --metric l2_rabitq --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/sift1M_HNSWBottomRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \


# MEX=48
# M=16
# efC=500
# bits=8
# taskset -c 1 test/search_hnsw_ngfix_rabitq \
# --is_rerank 1 --raw_vector_path /SSD/Text-to-Image/base.10M.fbin --q_bits 8 \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --metric ip_rabitq --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/t2i10M_HNSWNGFixRaBitQ_M${M}_efC${efC}_MEX${MEX}_${bits}bits.index \