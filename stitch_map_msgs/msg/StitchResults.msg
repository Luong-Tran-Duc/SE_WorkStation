# Output path
string output_path

# Number of input files processed
uint32 input_file_count

# Number of output files generated
uint32 output_file_count

# Stitch status
# 0: INIT
# 1: RUNNING
# 2: SUCCESS
# 3: FAILED
uint8 state

# Error message if state == FAILED
string error_message

# Processing time in seconds
float32 processing_time